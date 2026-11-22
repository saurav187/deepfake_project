import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import Lock
from time import monotonic
from uuid import uuid4

from .detectors.base import AnalysisResult, Detector
from .errors import UNREADABLE, ApiError

logger = logging.getLogger('uvicorn.error.jobs')


@dataclass
class Job:
    job_id: str
    created_at: datetime
    expires_at: datetime
    filename: str
    data: bytes
    status: str = 'queued'
    result: AnalysisResult | None = None
    error: dict[str, str] | None = None
    future: Future | None = None

    def snapshot(self):
        return {
            'job_id': self.job_id, 'status': self.status, 'result': self.result,
            'error': self.error, 'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'progress': {'queued': 0, 'done': 1}.get(self.status),
        }


class Jobs:
    def __init__(self, detector: Detector, workers: int, max_queue: int, ttl: float):
        self.detector = detector
        self.max_queue = max_queue
        self.ttl = timedelta(seconds=ttl)
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='analysis')
        self.lock = Lock()
        self.jobs: dict[str, Job] = {}
        self.queued = 0

    def enqueue(self, data: bytes, filename: str):
        with self.lock:
            if self.queued >= self.max_queue:
                raise ApiError(503, 'busy', 'Analysis is busy. Try again shortly.')
            now = datetime.now(UTC)
            job = Job(uuid4().hex, now, now + self.ttl, filename, data)
            self.jobs[job.job_id] = job
            self.queued += 1
            job.future = self.pool.submit(self.run, job)
            return job.job_id

    def run(self, job: Job):
        with self.lock:
            if job.status != 'queued':
                return
            self.queued -= 1
            job.status = 'processing'
            data, filename = job.data, job.filename
            job.data, job.filename = b'', ''
        started = monotonic()
        result = None
        error = None
        try:
            result = self.detector.analyze(data, filename)
        except ValueError:
            error = {'code': 'unreadable', 'message': UNREADABLE}
        except Exception:  # noqa: BLE001
            error = {'code': 'analysis_failed', 'message': 'Analysis failed. Try again.'}
        finally:
            del data
        with self.lock:
            job.future = None
            if job.status == 'processing':
                job.status = 'done' if error is None else 'failed'
                job.result = result
                job.error = error
                job.expires_at = datetime.now(UTC) + self.ttl
                if error is not None:
                    logger.info('job_id=%s status=failed error_code=%s', job.job_id, error['code'])
                elif result is not None:
                    logger.info('job_id=%s duration_s=%.3f verdict=%s elapsed_s=%.3f',
                                job.job_id, result['duration_s'], result['verdict'], monotonic() - started)

    def lookup(self, job_id: str):
        job = self.jobs.get(job_id)
        if job and job.status not in ('queued', 'processing') and job.expires_at <= datetime.now(UTC):
            del self.jobs[job_id]
            job = None
        if job is None:
            raise ApiError(404, 'not_found', 'This analysis could not be found.')
        return job

    def get(self, job_id: str):
        with self.lock:
            return self.lookup(job_id).snapshot()

    def cancel(self, job_id: str):
        with self.lock:
            job = self.lookup(job_id)
            if job.status == 'queued':
                self.queued -= 1
                job.future.cancel()
            if job.status in ('queued', 'processing'):
                job.status = 'cancelled'
                job.data, job.filename = b'', ''
                job.expires_at = datetime.now(UTC) + self.ttl
                logger.info('job_id=%s status=cancelled error_code=cancelled', job.job_id)
            elif job.status != 'cancelled':
                del self.jobs[job_id]

    def purge(self):
        with self.lock:
            now = datetime.now(UTC)
            expired = [job_id for job_id, job in self.jobs.items()
                       if job.status not in ('queued', 'processing') and job.expires_at <= now]
            for job_id in expired:
                del self.jobs[job_id]

    def close(self):
        self.pool.shutdown(cancel_futures=True)
        with self.lock:
            self.jobs.clear()
