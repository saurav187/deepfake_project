import logging
import subprocess
from dataclasses import replace
from threading import Event
from time import monotonic, sleep

import pytest
from fastapi.testclient import TestClient

from app.backend.config import Settings
from app.backend.detectors.mock import MockDetector
from app.backend.main import create_app
from app.backend.media import temporary_audio


@pytest.fixture(scope='session')
def fixtures(tmp_path_factory):
    folder = tmp_path_factory.mktemp('generated')
    variants = {
        'sample.wav': (3, []), 'sample.mp3': (3, []), 'sample.m4a': (3, []),
        'sample.ogg': (3, ['-c:a', 'libopus']), 'sample.opus': (3, ['-c:a', 'libopus']),
        'sample.mp4': (3, ['-f', 'lavfi', '-i', 'color=size=32x32:rate=1', '-shortest']),
        'short.wav': (0.3, []), 'long.wav': (720, ['-ar', '8000']),
        'minute.wav': (60, []),
    }
    for name, (seconds, extra) in variants.items():
        subprocess.run(
            ['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
             'sine=frequency=440:sample_rate=16000', '-f', 'lavfi', '-i', 'anoisesrc=a=0.03:s=16000',
             *extra, '-filter_complex', '[0:a][1:a]amix=inputs=2',
             '-t', str(seconds), str(folder / name)], check=True,
        )
    (folder / 'corrupt.wav').write_bytes(b'THIS_IS_PRIVATE_AUDIO_CONTENT')
    return folder


@pytest.fixture
def client():
    with TestClient(create_app(Settings())) as test_client:
        yield test_client


def upload(client, fixtures, name):
    return client.post('/api/analyze', files={'file': (name, (fixtures / name).read_bytes())})


def finished(client, job_id):
    deadline = monotonic() + 3
    while monotonic() < deadline:
        job = client.get(f'/api/jobs/{job_id}').json()
        if job['status'] not in ('queued', 'processing'):
            return job
        sleep(0.01)
    pytest.fail('Job did not finish')


@pytest.mark.parametrize('ext', ['wav', 'mp3', 'm4a', 'ogg', 'opus', 'mp4'])
def test_audio_formats(client, fixtures, ext):
    response = upload(client, fixtures, f'sample.{ext}')
    assert response.status_code == 202
    assert response.json()['status'] == 'queued'
    job = finished(client, response.json()['job_id'])
    assert job['status'] == 'done'
    assert job['progress'] == 1
    assert job['result']['duration_s'] == pytest.approx(3, abs=0.15)
    assert job['result']['verdict'] in {'likely_real', 'uncertain', 'likely_fake'}
    assert job['result']['segments']


@pytest.mark.parametrize('verdict', ['likely_real', 'uncertain', 'likely_fake', 'insufficient_audio'])
def test_forced_verdict(fixtures, monkeypatch, verdict):
    monkeypatch.setenv('MOCK_VERDICT', verdict)
    with TestClient(create_app(Settings())) as client:
        response = upload(client, fixtures, 'sample.wav')
        result = finished(client, response.json()['job_id'])['result']
    assert result['verdict'] == verdict
    assert (result['fake_probability'] is None) == (verdict == 'insufficient_audio')


def test_short(client, fixtures):
    response = upload(client, fixtures, 'short.wav')
    result = finished(client, response.json()['job_id'])['result']
    assert result['verdict'] == 'insufficient_audio'
    assert result['segments'] == []


@pytest.mark.parametrize(('name', 'code'), [('long.wav', 'too_long'), ('corrupt.wav', 'unreadable')])
def test_invalid_audio(client, fixtures, name, code):
    response = upload(client, fixtures, name)
    assert response.status_code == 422
    assert response.json()['error']['code'] == code


def test_unsupported_type(client):
    response = client.post('/api/analyze', files={'file': ('audio.txt', b'not audio')})
    assert response.status_code == 400
    assert response.json()['error']['code'] == 'unsupported_type'


def test_direct_60_mb_upload(client):
    response = client.post('/api/analyze', files={'file': ('large.wav', b'x' * 60_000_000)})
    assert response.status_code == 413
    assert response.json()['error']['code'] == 'too_large'


def test_streamed_oversize():
    def body():
        yield b'--test\r\nContent-Disposition: form-data; name="file"; filename="x.wav"\r\n\r\n'
        for _ in range(20):
            yield b'x' * 64_000
        yield b'\r\n--test--\r\n'

    with TestClient(create_app(Settings(max_upload_mb=1))) as client:
        response = client.post('/api/analyze', content=body(),
                               headers={'Content-Type': 'multipart/form-data; boundary=test'})
    assert response.status_code == 413


def test_queue_lifecycle_and_cancellation(fixtures, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger='uvicorn.error.jobs')
    started = Event()
    release = Event()
    analyze = MockDetector.analyze

    def blocked(self, data, filename):
        started.set()
        assert release.wait(5)
        return analyze(self, data, filename)

    monkeypatch.setattr(MockDetector, 'analyze', blocked)
    with TestClient(create_app(Settings(workers=1, max_queue=1))) as client:
        try:
            first = upload(client, fixtures, 'sample.wav').json()['job_id']
            assert started.wait(2)
            assert client.get(f'/api/jobs/{first}').json()['status'] == 'processing'

            second = upload(client, fixtures, 'sample.wav').json()['job_id']
            assert client.get(f'/api/jobs/{second}').json()['status'] == 'queued'
            busy = upload(client, fixtures, 'sample.wav')
            assert busy.status_code == 503
            assert busy.json()['error']['code'] == 'busy'
            assert client.delete(f'/api/jobs/{second}').status_code == 204
            assert client.get(f'/api/jobs/{second}').json()['status'] == 'cancelled'
            assert caplog.messages.count(f'job_id={second} status=cancelled error_code=cancelled') == 1

            release.set()
            assert finished(client, first)['status'] == 'done'
            assert client.delete(f'/api/jobs/{first}').status_code == 204
            assert client.get(f'/api/jobs/{first}').status_code == 404
        finally:
            release.set()


def test_ttl(fixtures):
    with TestClient(create_app(Settings(result_ttl_s=0.2))) as client:
        job_id = upload(client, fixtures, 'sample.wav').json()['job_id']
        assert finished(client, job_id)['status'] == 'done'
        deadline = monotonic() + 2
        store = client.app.state.jobs
        while monotonic() < deadline:
            with store.lock:
                if job_id not in store.jobs:
                    break
            sleep(0.05)
        else:
            pytest.fail('Background purge did not remove the expired job')
        assert client.get(f'/api/jobs/{job_id}').status_code == 404


@pytest.mark.parametrize('failure', [False, True])
def test_temp_cleanup_and_private_logs(client, fixtures, monkeypatch, tmp_path, caplog, failure):
    monkeypatch.setattr('tempfile.tempdir', str(tmp_path))
    caplog.set_level(logging.INFO)
    private = b'THIS_IS_PRIVATE_AUDIO_CONTENT'
    wav = (fixtures / 'sample.wav').read_bytes() + private

    if failure:
        def fail(self, data, filename):
            with temporary_audio(data, filename) as path:
                assert path.exists()
                raise RuntimeError(private.decode())

        monkeypatch.setattr(MockDetector, 'analyze', fail)

    response = client.post('/api/analyze', files={'file': ('sample.wav', wav)})
    job = finished(client, response.json()['job_id'])
    assert job['status'] == ('failed' if failure else 'done')
    if failure:
        assert job['error']['code'] == 'analysis_failed'
        assert f"job_id={job['job_id']} status=failed error_code=analysis_failed" in caplog.messages
    assert upload(client, fixtures, 'corrupt.wav').status_code == 422
    assert list(tmp_path.iterdir()) == []
    assert private.decode() not in caplog.text
    assert repr(wav[:64]) not in caplog.text


def test_completed_job_log(client, fixtures, caplog):
    caplog.set_level(logging.INFO, logger='uvicorn.error')
    job_id = upload(client, fixtures, 'sample.wav').json()['job_id']
    job = finished(client, job_id)
    assert job['status'] == 'done'
    result = job['result']
    prefix = f"job_id={job_id} duration_s={result['duration_s']:.3f} verdict={result['verdict']} elapsed_s="
    assert any(message.startswith(prefix) for message in caplog.messages)


def test_model_health_and_headers(client):
    response = client.get('/api/model')
    assert response.json()['model']['experimental']
    assert response.json()['limits']['max_upload_mb'] == 50
    assert response.headers['X-Content-Type-Options'] == 'nosniff'
    assert response.headers['Referrer-Policy'] == 'no-referrer'
    assert "default-src 'self'" in response.headers['Content-Security-Policy']
    assert client.get('/api/health').json() == {'status': 'ok', 'model_loaded': True}


def test_missing_detector(fixtures, caplog):
    with TestClient(create_app(replace(Settings(), backend='detector'))) as client:
        assert client.get('/api/health').json() == {'status': 'ok', 'model_loaded': False}
        response = upload(client, fixtures, 'sample.wav')
    assert response.status_code == 503
    assert response.json()['error']['code'] == 'model_unavailable'
    assert 'requires detector/inference.py' in caplog.text


def test_detector_load_exception(fixtures, monkeypatch, caplog):
    def fail(settings):
        raise RuntimeError('Could not initialize test device')

    monkeypatch.setattr('app.backend.main.load_backend', fail)
    with TestClient(create_app(Settings())) as client:
        assert client.get('/api/health').json() == {'status': 'ok', 'model_loaded': False}
        response = upload(client, fixtures, 'sample.wav')
    assert response.status_code == 503
    assert response.json()['error']['code'] == 'model_unavailable'
    assert 'RuntimeError: Could not initialize test device' in caplog.text
    assert 'Traceback (most recent call last):' in caplog.text


def test_minute_mock_performance(client, fixtures):
    started = monotonic()
    response = upload(client, fixtures, 'minute.wav')
    assert monotonic() - started < 1
    assert finished(client, response.json()['job_id'])['status'] == 'done'
    assert monotonic() - started < 2
