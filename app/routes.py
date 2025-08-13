# app/routes.py
# ✅ 모든 엔드포인트 모음 (Flask Blueprint)
from __future__ import annotations

import io
import re
import threading

import numpy as np
import torch
from PIL import Image, ImageOps
from flask import Blueprint, request, jsonify
from flask import Response
from transformers import TextIteratorStreamer

from cache import CACHE, make_key
from pipelines import get_ocr_easyocr
from pipelines import (
    get_text_generation, get_sentiment, get_translate_en2ko,
    get_summarize_en, get_summarize_ko, get_stream_gen
)
from utils import guard_image_bytes, parse_crop
from utils import guard_len

bp = Blueprint("api", __name__)


def _json():
    if request.is_json:
        body = request.get_json(silent=True) or {}
    else:
        body = {}
    return body


@bp.get("/health")
def health():
    return jsonify({
        "ok": True,
        "torch": torch.__version__,
        "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    })


@bp.post("/generate")
def generate():
    body = _json()
    prompt = str(body.get("prompt", ""))
    max_new_tokens = int(body.get("max_new_tokens", 60))
    temperature = float(body.get("temperature", 0.8))

    guard_len(prompt)
    pipe = get_text_generation()
    out = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        truncation=True,
        pad_token_id=pipe.tokenizer.pad_token_id,
    )[0]["generated_text"]
    return jsonify({"result": out})


@bp.post("/sentiment")
def sentiment():
    body = _json()
    text = str(body.get("text", ""))
    res = get_sentiment()(text)[0]
    return jsonify({"label": res["label"], "score": float(res["score"])})


@bp.post("/translate/en2ko")
def translate_en2ko():
    body = _json()
    texts = body.get("texts") or []
    num_beams = int(body.get("num_beams", 4))
    max_length = body.get("max_length", None)
    min_length = body.get("min_length", None)

    for t in texts:
        guard_len(t)

    key = make_key(
        "tr-en2ko",
        texts,
        num_beams,
        max_length if max_length is not None else "auto",
        min_length if min_length is not None else 0,
    )
    hit = CACHE.get(key)
    if hit is not None:
        return jsonify({"translations": hit, "cached": True})

    pipe = get_translate_en2ko()
    gen_kwargs = {"num_beams": num_beams}
    if max_length is not None:
        gen_kwargs["max_length"] = int(max_length)
    if min_length is not None:
        gen_kwargs["min_length"] = int(min_length)

    outs = pipe(texts, **gen_kwargs)
    result = [o["translation_text"] for o in outs]
    CACHE.put(key, result)
    return jsonify({"translations": result, "cached": False})


@bp.post("/summarize/en")
def summarize_en():
    body = _json()
    text = str(body.get("text", ""))
    max_length = int(body.get("max_length", 80))
    min_length = int(body.get("min_length", 30))
    num_beams = int(body.get("num_beams", 4))
    no_repeat_ngram_size = int(body.get("no_repeat_ngram_size", 3))

    guard_len(text)
    key = make_key("sum-en", text, max_length, min_length, num_beams, no_repeat_ngram_size)
    hit = CACHE.get(key)
    if hit is not None:
        return jsonify({"summary": hit, "cached": True})

    out = get_summarize_en()(
        text,
        max_length=max_length, min_length=min_length,
        do_sample=False, num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        truncation=True,
    )[0]["summary_text"]
    CACHE.put(key, out)
    return jsonify({"summary": out, "cached": False})


@bp.post("/summarize/ko")
def summarize_ko():
    body = _json()
    text = str(body.get("text", ""))
    max_length = int(body.get("max_length", 80))
    min_length = int(body.get("min_length", 30))
    num_beams = int(body.get("num_beams", 4))
    no_repeat_ngram_size = int(body.get("no_repeat_ngram_size", 3))

    guard_len(text)
    key = make_key("sum-ko", text, max_length, min_length, num_beams, no_repeat_ngram_size)
    hit = CACHE.get(key)
    if hit is not None:
        return jsonify({"summary": hit, "cached": True})

    out = get_summarize_ko()(
        text,
        max_length=max_length, min_length=min_length,
        do_sample=False, num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        truncation=True,
    )[0]["summary_text"]
    CACHE.put(key, out)
    return jsonify({"summary": out, "cached": False})


@bp.post("/generate/stream")
def generate_stream():
    """토큰이 생성되는 대로 텍스트를 흘려보내는 심플 스트림"""
    body = _json()
    prompt = str(body.get("prompt", ""))
    max_new_tokens = int(body.get("max_new_tokens", 60))
    temperature = float(body.get("temperature", 0.8))

    model, tok = get_stream_gen()
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    inputs = tok(prompt, return_tensors="pt")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        inputs = {k: v.to("mps") for k, v in inputs.items()}
        model = model.to("mps")

    th = threading.Thread(target=model.generate, kwargs=dict(
        **inputs, streamer=streamer, max_new_tokens=max_new_tokens,
        do_sample=True, temperature=temperature, pad_token_id=tok.eos_token_id
    ), daemon=True)
    th.start()

    def token_stream():
        for piece in streamer:
            yield piece

    return Response(token_stream(), mimetype="text/plain")


@bp.post("/ocr/id")
def ocr_id():
    """
    업로드 방식: multipart/form-data
      - field name: file  (jpg/png 등 이미지)
    query:
      - parse: bool       (기본 false)
      - crop: 'x,y,w,h'   (선택 크롭)
    """
    ctype = request.content_type or ""
    if "multipart/form-data" not in ctype:
        return jsonify({"error": {"type": "http", "message": "Content-Type must be multipart/form-data"}}), 415

    if "file" not in request.files:
        return jsonify({"error": {"type": "http", "message": "file is required"}}), 400

    data = request.files["file"].read()
    if not data:
        return jsonify({"error": {"type": "http", "message": "empty file"}}), 400

    # 크기 가드(기본 5MB, OCR_MAX_IMAGE_BYTES로 변경 가능)
    guard_image_bytes(data)

    # (선택) HEIC 지원
    try:
        from pillow_heif import register_heif_opener  # pip install pillow-heif
        register_heif_opener()
    except Exception:
        pass

    # 이미지 열기 + EXIF 회전 보정 + RGB
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        return jsonify({"error": {"type": "http", "message": "이미지 파일을 열 수 없습니다."}}), 400

    # 선택: crop 적용
    crop = request.args.get("crop")
    if crop:
        try:
            box = parse_crop(crop)
            img = img.crop(box)
        except Exception as e:
            return jsonify({"error": {"type": "http", "message": str(e)}}), 400

    # 너무 큰 이미지는 긴 변 1280으로 축소(속도/가독 개선)
    long_side = max(img.size)
    if long_side > 1280:
        s = 1280 / long_side
        img = img.resize((int(img.width * s), int(img.height * s)))

    # EasyOCR 실행
    reader = get_ocr_easyocr()
    lines = reader.readtext(np.array(img), detail=0)
    text = " ".join(lines)

    # (옵션) 간단 텍스트 파싱
    parse_flag = request.args.get("parse", "false").lower() == "true"

    def _parse_id_fields(text: str):
        def find(pat, flags=0):
            m = re.search(pat, text, flags)
            return m.group(1).strip() if m else None

        name = find(r"(?:Name|Full\s*Name)\s*[:\-]?\s*([A-Z][A-Za-z\s\.'\-]{1,40})")
        dob = find(r"(?:DOB|Birth\s*Date)\s*[:\-]?\s*([0-9]{2,4}[-./][0-9]{1,2}[-./][0-9]{1,2})")
        idno = find(r"(?:ID\s?(?:No|#)?|Identification(?:\s*No)?)\s*[:\-#]?\s*([A-Z0-9\-]{6,})")
        if idno and len(idno) > 4:
            import re as _re
            idno_masked = _re.sub(r"(?<=..).(?=..)", "*", idno)
        else:
            idno_masked = idno
        return {"name": name, "dob": dob, "id_number_masked": idno_masked}

    res = {"text": text, "backend": "easyocr"}
    if parse_flag:
        res["fields"] = _parse_id_fields(text)
    return jsonify(res)
