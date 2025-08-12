# app/routes.py
# ✅ 모든 엔드포인트 모음 (router)

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import torch
from transformers import TextIteratorStreamer
import threading

from schemas import GenerateReq, SentimentReq, TranslateReq, SummarizeReq, StreamGenerateReq
from pipelines import (
    get_text_generation, get_sentiment, get_translate_en2ko,
    get_summarize_en, get_summarize_ko, get_stream_gen
)
from config import guard_len
from cache import CACHE, make_key

router = APIRouter()

@router.get("/health")
def health():
    return {
        "ok": True,
        "torch": torch.__version__,
        "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }

@router.post("/generate")
def generate(req: GenerateReq):
    guard_len(req.prompt)
    pipe = get_text_generation()
    out = pipe(
        req.prompt,
        max_new_tokens=req.max_new_tokens,
        do_sample=True,
        temperature=req.temperature,
        truncation=True,
        pad_token_id=pipe.tokenizer.pad_token_id,
    )[0]["generated_text"]
    return {"result": out}

@router.post("/sentiment")
def sentiment(req: SentimentReq):
    res = get_sentiment()(req.text)[0]
    return {"label": res["label"], "score": float(res["score"])}

@router.post("/translate/en2ko")
def translate_en2ko(req: TranslateReq):
    # 1) 입력 길이 가드
    for t in req.texts:
        guard_len(t)

    # 2) 캐시 키 (None을 명시 문자열/값으로 치환하여 키 안정성 확보)
    key = make_key(
        "tr-en2ko",
        req.texts,
        req.num_beams,
        req.max_length if req.max_length is not None else "auto",  # None→"auto"
        req.min_length if req.min_length is not None else 0,       # None→0
    )
    hit = CACHE.get(key)
    if hit is not None:
        return {"translations": hit, "cached": True}

    # 3) 파이프라인 획득
    pipe = get_translate_en2ko()

    # 4) ✅ None은 kwargs에 넣지 말고, 제공된 값만 전달
    gen_kwargs = {"num_beams": req.num_beams}  # 품질/일관성용 빔서치
    if req.max_length is not None:
        gen_kwargs["max_length"] = req.max_length
    if req.min_length is not None:
        gen_kwargs["min_length"] = req.min_length

    # 5) 추론 실행
    outs = pipe(req.texts, **gen_kwargs)  # [{'translation_text': '...'}, ...]
    result = [o["translation_text"] for o in outs]

    # 6) 캐싱 후 반환
    CACHE.put(key, result)
    return {"translations": result, "cached": False}

@router.post("/summarize/en")
def summarize_en(req: SummarizeReq):
    guard_len(req.text)
    key = make_key("sum-en", req.text, req.max_length, req.min_length, req.num_beams, req.no_repeat_ngram_size)
    hit = CACHE.get(key)
    if hit is not None:
        return {"summary": hit, "cached": True}
    out = get_summarize_en()(
        req.text,
        max_length=req.max_length, min_length=req.min_length,
        do_sample=False, num_beams=req.num_beams,
        no_repeat_ngram_size=req.no_repeat_ngram_size,
        truncation=True,
    )[0]["summary_text"]
    CACHE.put(key, out)
    return {"summary": out, "cached": False}

@router.post("/summarize/ko")
def summarize_ko(req: SummarizeReq):
    guard_len(req.text)
    key = make_key("sum-ko", req.text, req.max_length, req.min_length, req.num_beams, req.no_repeat_ngram_size)
    hit = CACHE.get(key)
    if hit is not None:
        return {"summary": hit, "cached": True}
    out = get_summarize_ko()(
        req.text,
        max_length=req.max_length, min_length=req.min_length,
        do_sample=False, num_beams=req.num_beams,
        no_repeat_ngram_size=req.no_repeat_ngram_size,
        truncation=True,
    )[0]["summary_text"]
    CACHE.put(key, out)
    return {"summary": out, "cached": False}

@router.post("/generate/stream")
def generate_stream(req: StreamGenerateReq):
    """토큰이 생성되는 대로 텍스트를 흘려보내는 심플 스트림"""
    model, tok = get_stream_gen()
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    inputs = tok(req.prompt, return_tensors="pt")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        inputs = {k: v.to("mps") for k, v in inputs.items()}
        model = model.to("mps")

    th = threading.Thread(target=model.generate, kwargs=dict(
        **inputs, streamer=streamer, max_new_tokens=req.max_new_tokens,
        do_sample=True, temperature=req.temperature, pad_token_id=tok.eos_token_id
    ), daemon=True)
    th.start()

    def token_stream():
        for piece in streamer:
            yield piece

    return StreamingResponse(token_stream(), media_type="text/plain")
