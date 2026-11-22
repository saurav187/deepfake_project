import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from .config import EXTENSIONS, Settings
from .detectors.registry import load_backend
from .errors import UNREADABLE, ApiError
from .jobs import Jobs
from .media import probe_duration
from .uploads import read_upload

DIST = Path(__file__).resolve().parents[1] / 'frontend' / 'dist'
logger = logging.getLogger('uvicorn.error')


def create_app(settings: Settings | None = None):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        app.state.detector = None
        app.state.jobs = None
        try:
            detector = load_backend(settings)
            device = getattr(detector, 'device', settings.device)
            workers = settings.workers or (1 if str(device).startswith('cuda') else 2)
            app.state.jobs = Jobs(detector, workers, settings.max_queue, settings.result_ttl_s)
            app.state.detector = detector
        except Exception:
            logger.exception('The configured detector could not be loaded.')

        async def purge():
            while True:
                await asyncio.sleep(min(60, settings.result_ttl_s))
                app.state.jobs.purge()

        task = asyncio.create_task(purge()) if app.state.jobs else None
        try:
            yield
        finally:
            if task:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                await run_in_threadpool(app.state.jobs.close)

    app = FastAPI(lifespan=lifespan)
    if settings.allowed_origin:
        app.add_middleware(CORSMiddleware, allow_origins=[settings.allowed_origin],
                           allow_methods=['GET', 'POST', 'DELETE'], allow_headers=['Content-Type'])

    @app.middleware('http')
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; media-src 'self' blob:; object-src 'none'; frame-ancestors 'none'"
        )
        if request.url.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.exception_handler(ApiError)
    async def api_error(request, error):
        return JSONResponse({'error': error.error}, status_code=error.status)

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        code = 'not_found' if error.status_code == 404 else 'http_error'
        return JSONResponse({'error': {'code': code, 'message': str(error.detail)}},
                            status_code=error.status_code, headers=error.headers)

    def require_model():
        if app.state.detector is None:
            raise ApiError(503, 'model_unavailable', 'The analysis model is unavailable.')

    @app.get('/api/health')
    async def health():
        return {'status': 'ok', 'model_loaded': app.state.detector is not None}

    @app.get('/api/model')
    async def model():
        require_model()
        return {
            'model': app.state.detector.info,
            'limits': {'max_upload_mb': settings.max_upload_mb,
                       'max_duration_s': settings.max_duration_s,
                       'accepted_extensions': EXTENSIONS},
        }

    @app.post('/api/analyze', status_code=202)
    async def analyze(request: Request):
        require_model()
        data, filename = await read_upload(request, settings.max_upload_mb)
        if Path(filename).suffix.lower() not in EXTENSIONS:
            raise ApiError(400, 'unsupported_type', UNREADABLE)
        try:
            duration = await run_in_threadpool(probe_duration, data, filename)
        except (ValueError, OSError):
            raise ApiError(422, 'unreadable', UNREADABLE) from None
        if duration > settings.max_duration_s:
            raise ApiError(422, 'too_long',
                           f'Choose a recording that is {settings.max_duration_s / 60:g} minutes or shorter.')
        job_id = app.state.jobs.enqueue(data, filename)
        return {'job_id': job_id, 'status': 'queued'}

    @app.get('/api/jobs/{job_id}')
    async def get_job(job_id: str):
        require_model()
        return app.state.jobs.get(job_id)

    @app.delete('/api/jobs/{job_id}', status_code=204)
    async def delete_job(job_id: str):
        require_model()
        app.state.jobs.cancel(job_id)
        return Response(status_code=204)

    if DIST.is_dir():
        app.mount('/', StaticFiles(directory=DIST, html=True), name='frontend')
    return app


app = create_app()
