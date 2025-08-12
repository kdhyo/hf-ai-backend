# app/schemas.py
# ✅ 요청/응답 스키마

from pydantic import BaseModel, Field
from typing import List, Optional

class GenerateReq(BaseModel):
    prompt: str = Field(..., description="시작 프롬프트")
    max_new_tokens: int = Field(60, ge=1, le=512)
    temperature: float = Field(0.8, ge=0.0, le=2.0)

class SentimentReq(BaseModel):
    text: str

class TranslateReq(BaseModel):
    texts: List[str]
    num_beams: int = Field(4, ge=1, le=8)
    max_length: Optional[int] = Field(None, ge=16, le=512)
    min_length: Optional[int] = Field(None, ge=1, le=511)

class SummarizeReq(BaseModel):
    text: str
    max_length: int = Field(80, ge=16, le=512)
    min_length: int = Field(30, ge=1, le=511)
    num_beams: int = Field(4, ge=1, le=8)
    no_repeat_ngram_size: int = Field(3, ge=1, le=10)

class StreamGenerateReq(BaseModel):
    prompt: str
    max_new_tokens: int = Field(60, ge=1, le=512)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
