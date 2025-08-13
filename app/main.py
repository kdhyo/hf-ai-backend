# app/main.py — OpenAPI 3 (flask-openapi3)
import os
from flask_openapi3 import OpenAPI, Info, Tag
from middleware import install_middlewares, install_error_handlers  # 너의 Flask용 미들웨어 함수
from config import cleanup_mps_cache
from routes import api  # APIBlueprint 인스턴스

info = Info(
    title="hf-ai-backend",
    version="1.0.0",
    description="Hugging Face 모델들을 Flask로 감싼 데모 백엔드",
)

# OpenAPI 앱 생성 (+ Swagger/Redoc/RapiDoc UI는 extras 설치 시 자동 활성)
app = OpenAPI(__name__, info=info)

# (선택) 공통 태그 정의 예시 — 실제 라우트에서 tags=[nlp_tag] 처럼 사용 가능
nlp_tag = Tag(name="nlp", description="NLP endpoints")
ocr_tag = Tag(name="ocr", description="OCR endpoints")

# 미들웨어/에러 핸들러
install_middlewares(app)
install_error_handlers(app)

# ✅ APIBlueprint 등록 (register_blueprint가 아니라 register_api!)
app.register_api(api)

# 종료 훅: 캐시 비우기
@app.teardown_appcontext
def _on_teardown(exc):
    try:
        from pipelines import PIPELINES
        PIPELINES.clear()
    except Exception:
        pass
    cleanup_mps_cache()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="127.0.0.1", port=port, debug=True)
