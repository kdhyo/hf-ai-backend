# app/routes.py — APIBlueprint + Pydantic 타입힌트로 OAS3 자동생성
from __future__ import annotations

import io
import re
from typing import Optional

import torch
from PIL import Image, ImageOps
import numpy as np
from flask import Response, stream_with_context
from flask_openapi3 import APIBlueprint, Tag, FileStorage
from pydantic import BaseModel, Field
from transformers import TextIteratorStreamer

from cache import CACHE, make_key
from pipelines import (
    get_text_generation, get_sentiment, get_translate_en2ko,
    get_summarize_en, get_summarize_ko, get_stream_gen, get_ocr_easyocr
)
from schemas import GenerateReq, SentimentReq, TranslateReq, SummarizeReq, StreamGenerateReq
from utils import guard_len, guard_image_bytes, parse_crop

api = APIBlueprint("api", __name__, url_prefix="/")

nlp_tag = Tag(name="nlp", description="NLP endpoints")
misc_tag = Tag(name="misc", description="Misc/health")
ocr_tag = Tag(name="ocr", description="OCR endpoints")


# --------- 공용 모델 (Query/Form) ---------

class OCRQuery(BaseModel):
    parse: Optional[bool] = Field(default=False, description="간단 필드 파싱")
    crop: Optional[str] = Field(default=None, description="x,y,w,h")


class OCRForm(BaseModel):
    file: FileStorage


# --------- 엔드포인트들 ---------

@api.get("/health", tags=[misc_tag])
def health():
    """Health check"""
    return {
        "ok": True,
        "torch": torch.__version__,
        "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }


@api.post("/generate", tags=[nlp_tag])
def generate(body: GenerateReq):
    guard_len(body.prompt)
    pipe = get_text_generation()
    out = pipe(
        body.prompt,
        max_new_tokens=body.max_new_tokens,
        do_sample=True,
        temperature=body.temperature,
        truncation=True,
        pad_token_id=pipe.tokenizer.pad_token_id,
    )[0]["generated_text"]
    return {"result": out}


@api.post("/sentiment", tags=[nlp_tag])
def sentiment(body: SentimentReq):
    res = get_sentiment()(body.text)[0]
    return {"label": res["label"], "score": float(res["score"])}


@api.post("/translate/en2ko", tags=[nlp_tag])
def translate_en2ko(body: TranslateReq):
    # 입력 길이 가드 + 캐시 키
    for t in body.texts:
        guard_len(t)
    key = make_key("tr-en2ko", body.texts, body.num_beams,
                   body.max_length if body.max_length is not None else "auto",
                   body.min_length if body.min_length is not None else 0)
    hit = CACHE.get(key)
    if hit is not None:
        return {"translations": hit, "cached": True}

    pipe = get_translate_en2ko()
    gen_kwargs = {"num_beams": body.num_beams}
    if body.max_length is not None:
        gen_kwargs["max_length"] = body.max_length
    if body.min_length is not None:
        gen_kwargs["min_length"] = body.min_length

    outs = pipe(body.texts, **gen_kwargs)
    result = [o["translation_text"] for o in outs]
    CACHE.put(key, result)
    return {"translations": result, "cached": False}


@api.post("/summarize/en", tags=[nlp_tag])
def summarize_en(body: SummarizeReq):
    guard_len(body.text)
    key = make_key("sum-en", body.text, body.max_length, body.min_length, body.num_beams, body.no_repeat_ngram_size)
    hit = CACHE.get(key)
    if hit is not None:
        return {"summary": hit, "cached": True}
    out = get_summarize_en()(
        body.text,
        max_length=body.max_length, min_length=body.min_length,
        do_sample=False, num_beams=body.num_beams,
        no_repeat_ngram_size=body.no_repeat_ngram_size,
        truncation=True,
    )[0]["summary_text"]
    CACHE.put(key, out)
    return {"summary": out, "cached": False}


@api.post("/summarize/ko", tags=[nlp_tag])
def summarize_ko(body: SummarizeReq):
    guard_len(body.text)
    key = make_key("sum-ko", body.text, body.max_length, body.min_length, body.num_beams, body.no_repeat_ngram_size)
    hit = CACHE.get(key)
    if hit is not None:
        return {"summary": hit, "cached": True}
    out = get_summarize_ko()(
        body.text,
        max_length=body.max_length, min_length=body.min_length,
        do_sample=False, num_beams=body.num_beams,
        no_repeat_ngram_size=body.no_repeat_ngram_size,
        truncation=True,
    )[0]["summary_text"]
    CACHE.put(key, out)
    return {"summary": out, "cached": False}


@api.post(
    "/generate/stream",
    tags=[nlp_tag],
    # OAS3 응답 스펙: text/plain 스트림
    responses={200: {"content": {"text/plain": {"schema": {"type": "string"}}}}}
)
def generate_stream(body: StreamGenerateReq):
    """토큰 생성 즉시 흘려보내는 심플 스트림(text/plain)"""
    model, tok = get_stream_gen()
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    inputs = tok(body.prompt, return_tensors="pt")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        inputs = {k: v.to("mps") for k, v in inputs.items()}
        model = model.to("mps")

    import threading
    th = threading.Thread(target=model.generate, kwargs=dict(
        **inputs, streamer=streamer, max_new_tokens=body.max_new_tokens,
        do_sample=True, temperature=body.temperature, pad_token_id=tok.eos_token_id
    ), daemon=True)
    th.start()

    def _gen():
        for piece in streamer:
            yield piece

    return Response(stream_with_context(_gen()), mimetype="text/plain")


# ---------- OCR (multipart/form-data) ----------
@api.post("/ocr/id", tags=[ocr_tag])
def ocr_id(form: OCRForm, query: OCRQuery):
    """
    파일은 form(file), 옵션은 query(parse/crop)
    """
    stream = form.file.stream.read()
    guard_image_bytes(stream)

    # PIL로 열고 EXIF 회전 보정 + RGB
    try:
        img = Image.open(io.BytesIO(stream))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        return {"error": {"type": "http", "message": "이미지 파일을 열 수 없습니다."}}, 400

    # 선택 크롭
    box = parse_crop(query.crop) if query.crop else None
    if box:
        img = img.crop(box)

    # 너무 크면 긴 변 1280으로 축소(속도/가독)
    long_side = max(img.size)
    if long_side > 1280:
        s = 1280 / long_side
        img = img.resize((int(img.width * s), int(img.height * s)))

    # ✅ EasyOCR 입력: numpy 배열로 변환
    np_img = np.array(img)

    reader = get_ocr_easyocr()
    try:
        # detail=0 -> 텍스트만 리스트, paragraph=True는 문장 단위 합침 경향
        lines = reader.readtext(np_img, detail=0, paragraph=True)
    except Exception as e:
        return {"error": {"type": "ocr", "message": str(e)}}, 400

    text = "\n".join(lines).strip()

    result = {"text": text, "backend": "easyocr"}
    if query.parse:
        def _parse_id_fields(text: str):
            def find(pat, flags=0):
                m = re.search(pat, text, flags)
                return m.group(1).strip() if m else None
            name = find(r"(?:Name|Full\s*Name)[:\-]?\s*([A-Z][A-Za-z\s\.'\-]{1,40})")
            dob  = find(r"(?:DOB|Birth\s*Date)[:\-]?\s*([0-9]{2,4}[-./][0-9]{1,2}[-./][0-9]{1,2})")
            idno = find(r"(?:ID\s?(?:No|#)?|Identification(?:\s*No)?)[:\-#]?\s*([A-Z0-9\-]{6,})")
            if idno and len(idno) > 4:
                import re as _re
                idno_masked = _re.sub(r"(?<=..).(?=..)", "*", idno)
            else:
                idno_masked = idno
            return {"name": name, "dob": dob, "id_number_masked": idno_masked}
        result["fields"] = _parse_id_fields(text)

    return result

