# app/config.py
# ✅ 환경/디바이스/경고 설정 (Flask용 예외)

import os, importlib, gc
import torch
import warnings
from transformers.utils import logging as hf_logging
from werkzeug.exceptions import RequestEntityTooLarge

# ▶ 경고 억제용 환경변수
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ▶ Transformers 경고/로그 톤 다운
warnings.filterwarnings("ignore", message=r"You passed `num_labels=.*")
hf_logging.set_verbosity_error()

# ▶ 입력 길이 상한(문자)
MAX_CHARS = 4000

def guard_len(text: str):
    """요청 본문 길이 가드"""
    if text is None:
        return
    if len(text) > MAX_CHARS:
        raise RequestEntityTooLarge(description=f"Input too long (> {MAX_CHARS} chars)")

def device_kwargs() -> dict:
    """
    accelerate가 있으면 device_map='auto' (GPU/MPS 사용),
    없으면 안전하게 CPU 폴백.
    """
    has_acc = importlib.util.find_spec("accelerate") is not None
    kw = {}
    if has_acc:
        kw["device_map"] = "auto"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            kw["torch_dtype"] = torch.float16
    return kw

def cleanup_mps_cache():
    """앱 종료 시 MPS 캐시/GC"""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

def local_only() -> bool:
    """HF_HUB_OFFLINE=1 이면 네트워크 금지 + 캐시만 사용."""
    return os.getenv("HF_HUB_OFFLINE", "0") == "1"
