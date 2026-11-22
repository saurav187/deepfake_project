import importlib
import subprocess

import pytest

from app.backend.detectors.legacy import CHECKPOINT, PIPELINE, LegacyDetector

pytestmark = pytest.mark.skipif(
    not PIPELINE.is_dir() or not CHECKPOINT.is_file(),
    reason='The frozen legacy pipeline or checkpoint is absent',
)


@pytest.fixture(scope='module')
def legacy():
    return LegacyDetector('cpu')


@pytest.mark.parametrize(('filename', 'output_options'), [
    ('stereo.wav', []),
    ('mono.wav', ['-ar', '16000', '-ac', '1']),
    ('sample.m4a', ['-c:a', 'aac']),
])
def test_decode_matches_frozen_samples(legacy, tmp_path, filename, output_options):
    import numpy as np

    path = tmp_path / filename
    subprocess.run(
        ['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
         'aevalsrc=0.3*sin(2*PI*440*t)|0.2*sin(2*PI*660*t):s=44100:d=0.3',
         *output_options, str(path)],
        check=True,
    )
    assert np.array_equal(legacy.decode(path), legacy.pipeline.read_audio(path))


def test_constructing_legacy_keeps_subprocess_binding(legacy):
    assert importlib.import_module('audio_features').subprocess is subprocess
