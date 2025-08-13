# app/main.py
# ✅ Flask 앱 생성/수명주기/미들웨어/블루프린트

import os
from flask import Flask
from middleware import install_middlewares, install_error_handlers
from routes import bp as api_bp
from config import cleanup_mps_cache

def create_app() -> Flask:
    app = Flask("hf-ai-backend")
    app.config.update(
        APP_TITLE="hf-ai-backend",
        APP_DESCRIPTION="Hugging Face 모델들을 Flask로 감싼 데모 백엔드",
        APP_VERSION="1.0.0",
    )

    # 미들웨어/에러핸들러/블루프린트 장착
    install_middlewares(app)
    install_error_handlers(app)
    app.register_blueprint(api_bp)

    # 수명주기: 종료 훅(개발 서버 기준)
    @app.teardown_appcontext
    def _on_teardown(exc):
        try:
            from pipelines import PIPELINES
            PIPELINES.clear()
        except Exception:
            pass
        cleanup_mps_cache()
    return app

app = create_app()

if __name__ == "__main__":
    # 개발 서버 실행 (배포는 gunicorn/uwsgi 권장)
    port = int(os.getenv("PORT", "8000"))
    app.run(host="127.0.0.1", port=port, debug=True)
