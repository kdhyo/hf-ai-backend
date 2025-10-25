from __future__ import annotations

import io
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageOps
from flask import Response, stream_with_context
from flask_openapi3 import APIBlueprint, Tag, FileStorage
from pydantic import BaseModel
from pydantic import Field
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


class KRIDFields(BaseModel):
    name: Optional[str] = None
    rrn_masked: Optional[str] = None
    issue_date: Optional[str] = None
    issuer: Optional[str] = None
    address: Optional[str] = None


def _normalize_kr_ocr(s: str) -> str:
    """한글 주민등록증에서 자주 나오는 OCR 오탈자 간단 보정"""
    rep = [
        (r"서용특별시", "서울특별시"),
        (r"미표구", "마포구"),
        (r"(?<=\d)중\b", "층"),  # 8중 -> 8층
        (r"\s{2,}", " "),  # 다중 공백 축소
    ]
    import re
    t = s
    for pat, repl in rep:
        t = re.sub(pat, repl, t)
    return t.strip()


def _parse_id_fields_kr(text: str) -> KRIDFields:
    """
    주민등록증 OCR 텍스트 파서(한글):
    - name: '주민등록증' 다음 또는 주민번호 이전 라인의 짧은 한글 이름(2~6자)
    - rrn_masked: XXXXXX-*******
    - issue_date: YYYY-MM-DD (예: '2018. 1. 28.' -> '2018-01-28')
    - issuer: '서울특별시 마포구청장' 등
    - address: 주민번호가 있는 라인에서 주민번호 뒤쪽, 없으면 후보 라인 중 주소 토큰 포함 & 발급기관/날짜 제외
    """
    import re
    # 원본 정리
    raw = re.sub(r"[^\S\r\n]+", " ", text).strip()
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    # 주민등록번호 탐색 (라인/인덱스 기록)
    rrn_masked, rrn_line_idx, rrn_span = None, None, None
    rrn_pat = re.compile(r"(?P<rrn>\d{6}[ -]?\d{7})")
    for i, ln in enumerate(lines):
        m = rrn_pat.search(ln)
        if m:
            rrn_raw = m.group("rrn").replace(" ", "").replace("-", "")
            rrn_masked = f"{rrn_raw[:6]}-" + "*" * 7
            rrn_line_idx = i
            rrn_span = m.span()
            break

    # 발급일
    issue_date = None
    m_date = re.search(r"((19|20)?\d{2})\.\s*(\d{1,2})\.\s*(\d{1,2})\.", raw)
    if m_date:
        y = m_date.group(1)
        m = int(m_date.group(3));
        d = int(m_date.group(4))
        if len(y) == 2:  # '18.' 같은 2자리 연도 → 2018/1918 추정
            y = "20" + y if int(y) < 50 else "19" + y
        issue_date = f"{int(y):04d}-{m:02d}-{d:02d}"

    # 이름
    name = None
    before_rrn_text = raw if rrn_line_idx is None else "\n".join(lines[:rrn_line_idx + 1])
    m_name = re.search(r"주민등록증\s*([가-힣]{2,6})", before_rrn_text)
    if m_name:
        name = m_name.group(1)
    else:
        # 숫자/한자/영문 거의 없는 짧은 한글 라인
        for ln in lines[: (rrn_line_idx if rrn_line_idx is not None else len(lines))]:
            if 2 <= len(ln) <= 6 and re.fullmatch(r"[가-힣]{2,6}", ln):
                name = ln
                break

    # 발급기관 (오탈 보정 후 탐색)
    norm_all = _normalize_kr_ocr(raw)
    m_issuer = re.search(r"([가-힣]{2,10}(특별시|광역시|도))\s*[가-힣]{1,3}구청장", norm_all)
    issuer = m_issuer.group(0) if m_issuer else None

    # 주소 추출
    address = None
    if rrn_line_idx is not None and rrn_span is not None:
        # 주민번호가 포함된 라인의 주민번호 '뒤쪽'을 1순위 주소 후보로 사용
        rrn_line = lines[rrn_line_idx]
        tail = rrn_line[rrn_span[1]:].strip()  # 주민번호 이후
        if tail:
            address = tail

    # 1순위 주소 후보가 없으면, 후보 라인에서 선택
    if not address:
        # 후보: 주민번호 라인 이후 2줄 우선 + 전체 라인
        candidates = []
        if rrn_line_idx is not None:
            candidates.extend(lines[rrn_line_idx + 1: rrn_line_idx + 3])
        candidates.extend(lines)

        addr_tokens1 = r"(특별시|광역시|도|시)"
        addr_tokens2 = r"(구|군|동|읍|면|로|길)"
        exclude_tokens = r"(청장|시장|군수|구청장|면장|읍장)"
        date_like = re.compile(r"(\d{2,4})\.\s*\d{1,2}\.\s*\d{1,2}\.")  # 2018. 1. 28.

        for ln in candidates:
            if date_like.search(ln):  # 날짜라인 제외
                continue
            if re.search(exclude_tokens, ln):  # 발급기관/직함 제외
                continue
            # 두 그룹 토큰을 모두 포함(전방탐색)해야 주소로 인정
            if re.search(addr_tokens1, ln) and re.search(addr_tokens2, ln):
                address = ln
                break

    # 주소 오탈 보정 + 공백 정리
    if address:
        address = _normalize_kr_ocr(address)

    return KRIDFields(
        name=name,
        rrn_masked=rrn_masked,
        issue_date=issue_date,
        issuer=issuer,
        address=address,
    )


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

    # EasyOCR 입력: numpy 배열로 변환
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
        fields = _parse_id_fields_kr(text)
        result["fields"] = fields.model_dump(exclude_none=True)

    return result
