# app/main.py
# ✅ FastAPI 앱 생성/수명주기/미들웨어/라우터

from contextlib import asynccontextmanager
from fastapi import FastAPI
from middleware import RequestLoggingMiddleware, http_exc_handler, unhandled_exc_handler
from routes import router
from pipelines import PIPELINES
from config import cleanup_mps_cache

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ▶ 앱 시작 시 (지금은 준비할 것 없음)
    yield
    # ▶ 앱 종료 시 (리소스 정리)
    PIPELINES.clear()
    cleanup_mps_cache()

app = FastAPI(
    title="hf-ai-backend",
    description="Hugging Face 모델들을 FastAPI로 감싼 데모 백엔드",
    version="1.0.0",
    lifespan=lifespan,
)

# 미들웨어/핸들러/라우터 장착
app.add_middleware(RequestLoggingMiddleware)
app.add_exception_handler(Exception, unhandled_exc_handler)
# HTTPException은 FastAPI가 기본 핸들러를 갖지만, 커스텀 JSON을 쓰려면 아래도 등록
from fastapi import HTTPException
app.add_exception_handler(HTTPException, http_exc_handler)
app.include_router(router)

# app/main.py 맨 아래 실행부 교체
if __name__ == "__main__":
    import uvicorn, sys, pathlib
    # ▶ 프로젝트 루트를 sys.path에 주입
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)

