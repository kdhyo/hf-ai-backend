# app/middleware.py
# ✅ 요청 로깅/Request-ID + 에러 JSON 통일 (Flask)

import time, uuid, logging
from flask import request, g, jsonify
from werkzeug.exceptions import HTTPException

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("hf-api")

def install_middlewares(app):
    @app.before_request
    def _before():
        rid = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        g.request_id = rid
        g._start_ts = time.perf_counter()

    @app.after_request
    def _after(response):
        # 로깅
        dur_ms = int((time.perf_counter() - getattr(g, "_start_ts", time.perf_counter())) * 1000)
        log.info(
            f'{{"lvl":"info","rid":"{getattr(g,"request_id",None)}","m":"{request.method}",'
            f'"p":"{request.path}","status":{response.status_code},"ms":{dur_ms}}}'
        )
        # 리스폰스 헤더에 Request-ID 부착
        if getattr(g, "request_id", None):
            response.headers["X-Request-ID"] = g.request_id
        return response

def install_error_handlers(app):
    @app.errorhandler(HTTPException)
    def _http_exc(e: HTTPException):
        rid = getattr(g, "request_id", None)
        return jsonify({
            "error": {
                "type": "http",
                "code": e.code,
                "name": e.name,
                "message": e.description,
                "request_id": rid
            }
        }), e.code

    @app.errorhandler(Exception)
    def _unhandled(e: Exception):
        rid = getattr(g, "request_id", None)
        log.error(f'{{"lvl":"error","rid":"{rid}","error":"{type(e).__name__}","msg":"{str(e)}"}}')
        return jsonify({
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "request_id": rid
            }
        }), 500
