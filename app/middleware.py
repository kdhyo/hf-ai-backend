# app/middleware.py
# ✅ 요청 로깅/Request-ID + 에러 JSON 통일

import time, uuid, logging
from starlette.requests import Request
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import HTTPException
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("hf-api")

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = req_id
        start = time.perf_counter()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            dur_ms = int((time.perf_counter() - start) * 1000)
            log.info(f'{{"lvl":"info","rid":"{req_id}","m":"{request.method}","p":"{request.url.path}","status":{getattr(response,"status_code",500)},"ms":{dur_ms}}}')

async def http_exc_handler(request: Request, exc: HTTPException):
    rid = getattr(request.state, "request_id", None)
    return JSONResponse(status_code=exc.status_code, content={"error": {"type":"http","message":exc.detail,"request_id":rid}})

async def unhandled_exc_handler(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", None)
    log.error(f'{{"lvl":"error","rid":"{rid}","error":"{type(exc).__name__}","msg":"{str(exc)}"}}')
    return JSONResponse(status_code=500, content={"error": {"type": type(exc).__name__, "message": str(exc), "request_id": rid}})
