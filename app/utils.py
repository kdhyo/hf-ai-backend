# app/utils.py
# ------------------------------------------------------------
# 공통 유틸: 텍스트 길이 가드 / 이미지 바이트 가드 / crop 파라미터 파서 (Flask)
# ------------------------------------------------------------
import os

from werkzeug.exceptions import RequestEntityTooLarge, BadRequest

# 텍스트 입력 최대 길이(문자 수) — 환경변수로 덮어쓰기 가능
MAX_CHARS = int(os.getenv("MAX_CHARS", 4000))


def guard_len(text: str, limit: int = MAX_CHARS) -> None:
    """문자열 길이가 limit를 넘으면 413 에러"""
    if text is None:
        return
    if len(text) > limit:
        raise RequestEntityTooLarge(description=f"Input too long (> {limit} chars)")


def guard_image_bytes(data: bytes, max_bytes: int = None) -> None:
    """이미지 원본 바이트 크기가 한도를 넘으면 413 에러"""
    max_bytes = max_bytes or int(os.getenv("OCR_MAX_IMAGE_BYTES", 5 * 1024 * 1024))  # 기본 5MB
    if len(data) > max_bytes:
        raise RequestEntityTooLarge(description=f"Image too large (> {max_bytes} bytes)")


def parse_crop(crop: str):
    """
    'x,y,w,h' 문자열을 (x1, y1, x2, y2) 튜플로 변환.
    잘못된 형식이면 400 에러.
    """
    if not crop:
        return None
    try:
        x, y, w, h = map(int, crop.split(","))
        if w <= 0 or h <= 0:
            raise ValueError("w/h must be positive")
        return (x, y, x + w, y + h)
    except Exception:
        raise BadRequest(description="crop must be 'x,y,w,h' integers")
