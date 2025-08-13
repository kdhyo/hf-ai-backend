# app/pipelines.py
# ✅ HF 파이프라인 로더 + 캐시 (게으른 로딩)

from typing import Any, Dict, Tuple

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from config import device_kwargs

# 전역 캐시들
PIPELINES: Dict[str, Any] = {}
STREAM_STATE: Dict[str, Tuple[Any, Any]] = {}


def get_text_generation():
    if "generate" not in PIPELINES:
        tok = AutoTokenizer.from_pretrained("distilgpt2")
        tok.pad_token = tok.eos_token
        PIPELINES["generate"] = pipeline(
            task="text-generation", model="distilgpt2", tokenizer=tok, **device_kwargs()
        )
    return PIPELINES["generate"]


def get_sentiment():
    if "sentiment" not in PIPELINES:
        PIPELINES["sentiment"] = pipeline(task="sentiment-analysis", **device_kwargs())
    return PIPELINES["sentiment"]


def get_translate_en2ko():
    # 권장: NLLB (언어코드 명시 → 출력 안정)
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    if "translate_en2ko" not in PIPELINES:
        model_id = "facebook/nllb-200-distilled-600M"
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        PIPELINES["translate_en2ko"] = pipeline(
            task="translation",
            model=model,
            tokenizer=tok,
            src_lang="eng_Latn",
            tgt_lang="kor_Hang",
            **device_kwargs(),
        )
    return PIPELINES["translate_en2ko"]


def get_summarize_en():
    if "summarize_en" not in PIPELINES:
        model_id = "facebook/bart-large-cnn"
        tok = AutoTokenizer.from_pretrained(model_id)
        tok.model_max_length = 1024
        PIPELINES["summarize_en"] = pipeline(
            task="summarization", model=model_id, tokenizer=tok, **device_kwargs()
        )
    return PIPELINES["summarize_en"]


def get_summarize_ko():
    if "summarize_ko" not in PIPELINES:
        model_id = "gogamza/kobart-summarization"
        tok = AutoTokenizer.from_pretrained(model_id)
        tok.model_max_length = 1024
        PIPELINES["summarize_ko"] = pipeline(
            task="summarization", model=model_id, tokenizer=tok, **device_kwargs()
        )
    return PIPELINES["summarize_ko"]


def get_stream_gen():
    """스트리밍용 (모델, 토크나이저) 1회 로드"""
    if "stream" not in STREAM_STATE:
        model_id = "distilgpt2"
        tok = AutoTokenizer.from_pretrained(model_id)
        tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id, **device_kwargs())
        STREAM_STATE["stream"] = (model, tok)
    return STREAM_STATE["stream"]


# --- OCR (EasyOCR) ---
def get_ocr_easyocr():
    """ko+en 일반 OCR용(텍스트 검출+인식). 설치: pip install easyocr"""
    if "ocr_easyocr" not in PIPELINES:
        import easyocr  # type: ignore
        PIPELINES["ocr_easyocr"] = easyocr.Reader(["ko", "en"])  # 한글+영어
    return PIPELINES["ocr_easyocr"]
