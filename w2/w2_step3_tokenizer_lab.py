# w2_step3_tokenizer_lab.py
# ------------------------------------------------------------
# 목적:
#  1) 토크나이저 기본기: 인코딩/디코딩, 패딩, 트렁케이션, 배치 인코딩, attention_mask
#  2) 스페셜 토큰: EOS/BOS/[CLS]/[SEP] 등 (모델마다 다름)
#  3) text pair 인코딩 (문장쌍 입력: BERT/BART 계열에서 자주 사용)
#  4) 길이 측정/청크 분할: 요약/번역 입력 토큰 길이 안전하게 자르기
#
# 전제:
#  - Step2에서 이미 다음 모델들을 캐시해둔 상태라고 가정:
#    * distilgpt2 (GPT-2 계열, decoder-only → pad_token 없음)
#    * facebook/bart-large-cnn (encoder-decoder → BOS/EOS 사용)
#    * gogamza/kobart-summarization (KoBART, 한국어 요약)
#
# 오프라인 모드:
#  - 환경변수 HF_HUB_OFFLINE=1 이면 캐시만 사용 (네트워크 차단)
#    예) export HF_HUB_OFFLINE=1
# ------------------------------------------------------------

import os
from typing import List, Tuple
from transformers import AutoTokenizer

# ✅ 오프라인이면 캐시만 사용
LOCAL_ONLY = os.getenv("HF_HUB_OFFLINE", "0") == "1"

def p(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

# ---------------------------------------------------------------------
# 0) 토크나이저 로드 (영어 GPT-2 / 영어 BART / 한국어 KoBART)
# ---------------------------------------------------------------------
p("0) 토크나이저 로드")

# GPT-2 계열은 pad_token이 없음 → EOS를 패딩으로 지정하는 패턴이 일반적
tok_gpt2 = AutoTokenizer.from_pretrained("distilgpt2", local_files_only=LOCAL_ONLY)
if tok_gpt2.pad_token is None:
    tok_gpt2.pad_token = tok_gpt2.eos_token  # 경고 억제/배치 패딩을 위해 지정

# BART(영어) — encoder-decoder. BOS(<s>), EOS(</s>) 등 스페셜 토큰 사용
tok_bart = AutoTokenizer.from_pretrained("facebook/bart-large-cnn", local_files_only=LOCAL_ONLY)

# KoBART(한국어) — 한국어 요약용, 토큰화 차이를 체감해보자
tok_kobart = AutoTokenizer.from_pretrained("gogamza/kobart-summarization", local_files_only=LOCAL_ONLY)

print("[GPT-2] pad_token:", tok_gpt2.pad_token, "| eos:", tok_gpt2.eos_token, "| max_len:", tok_gpt2.model_max_length)
print("[BART ] pad_token:", tok_bart.pad_token, "| bos:", tok_bart.bos_token, "| eos:", tok_bart.eos_token, "| max_len:", tok_bart.model_max_length)
print("[KoBART] pad_token:", tok_kobart.pad_token, "| bos:", tok_kobart.bos_token, "| eos:", tok_kobart.eos_token, "| max_len:", tok_kobart.model_max_length)

# ---------------------------------------------------------------------
# 1) 기본 인코딩/디코딩 (단건)
# ---------------------------------------------------------------------
p("1) 기본 인코딩/디코딩 (단건) - GPT-2")

text_en = "Hello Hugging Face! Tokenization turns text into ids."
enc = tok_gpt2(text_en)  # dict: input_ids, attention_mask(모델/옵션에 따라)
print("input_ids (len):", len(enc["input_ids"]))
print("tokens       :", tok_gpt2.convert_ids_to_tokens(enc["input_ids"])[:20])
print("decode       :", tok_gpt2.decode(enc["input_ids"]))

# ---------------------------------------------------------------------
# 2) 배치 인코딩 + 패딩/트렁케이션
# ---------------------------------------------------------------------
p("2) 배치 인코딩 + 패딩/트렁케이션 - GPT-2")

batch_texts = [
    "Short line.",
    "This is a slightly longer line that will show padding.",
]
# padding=True  → 가장 긴 문장 길이에 맞춰 패딩
# truncation=True → model_max_length 넘으면 잘라냄
# return_tensors="pt" → 파이토치 텐서 반환 (배치 추론에 바로 사용)
batch = tok_gpt2(batch_texts, padding=True, truncation=True, return_tensors="pt")
print("input_ids.shape   :", batch["input_ids"].shape)      # (batch, seq_len)
print("attention_mask    :", batch["attention_mask"][0][:20])
print("pad_token_id      :", tok_gpt2.pad_token_id)

# max_length를 작게 잡아 트렁케이션 체감
short = tok_gpt2(batch_texts, padding="max_length", truncation=True, max_length=10, return_tensors="pt")
print("max_length=10 shape:", short["input_ids"].shape)
print("row0 ids         :", short["input_ids"][0].tolist())
print("row0 tokens      :", tok_gpt2.convert_ids_to_tokens(short["input_ids"][0])[:15])

# ---------------------------------------------------------------------
# 3) 스페셜 토큰과 text pair (BART)
#   - BART/KoBART는 BOS(<s>), EOS(</s>), SEP 등 스페셜 토큰이 명시됨
#   - text_pair를 주면 두 문장을 하나의 시퀀스로 인코딩
# ---------------------------------------------------------------------
p("3) 스페셜 토큰 & text_pair - BART")

premise = "Hugging Face hosts models and datasets on the Hub."
hypo    = "You can download models programmatically in Python."

single = tok_bart(premise)
paired = tok_bart(premise, text_pair=hypo)  # 문장쌍

print("single tokens (head):", tok_bart.convert_ids_to_tokens(single["input_ids"][:15]))
print("paired tokens (head):", tok_bart.convert_ids_to_tokens(paired["input_ids"][:25]))
print("special_tokens_mask :", paired.get("special_tokens_mask", [])[:25])  # 스페셜 토큰 위치(1) 표시

# KoBART로도 같은 쌍을 인코딩해 차이 비교(토큰화/스페셜토큰 다름)
paired_ko = tok_kobart("허깅페이스는 허브에 모델과 데이터셋을 호스팅합니다.", text_pair="파이썬으로 프로그램 방식으로 모델을 다운로드할 수 있습니다.")
print("KoBART paired tokens (head):", tok_kobart.convert_ids_to_tokens(paired_ko["input_ids"][:25]))

# ---------------------------------------------------------------------
# 4) 길이 측정 & 청크 분할 (요약/번역 입력 안전화)
#   - 요약 모델(BART/KoBART)은 보통 입력 max 1024 토큰 내
#   - 긴 텍스트를 안전하게 쪼개서 처리하는 유틸 (overlap 포함)
# ---------------------------------------------------------------------
p("4) 길이 측정 & 청크 분할 유틸 - BART/KoBART 공통")

def count_tokens(tok, text: str) -> int:
    """텍스트를 토크나이즈했을 때 토큰 개수(스페셜 포함)를 반환."""
    return len(tok(text)["input_ids"])

def make_chunks(tok, text: str, max_tokens: int, overlap_tokens: int = 50) -> List[str]:
    """
    긴 텍스트를 토큰 기준으로 잘라내기.
    - max_tokens: 각 청크의 최대 토큰 수 (스페셜 포함 여유분까지 고려 권장)
    - overlap_tokens: 청크 사이에 겹치게 넣을 토큰 수 (문맥 이어짐 도움)
    """
    ids = tok(text)["input_ids"]
    chunks = []
    i = 0
    while i < len(ids):
        # 현재 청크 범위 결정
        end = min(i + max_tokens, len(ids))
        piece_ids = ids[i:end]
        chunks.append(tok.decode(piece_ids, skip_special_tokens=False))
        if end == len(ids):
            break
        # 다음 시작 인덱스: 겹치기(overlap)만큼 뒤로 물러서 진행
        i = end - overlap_tokens if end - overlap_tokens > i else end
    return chunks

long_text = (
    "Hugging Face provides an open platform for sharing machine learning models and datasets. "
    "Developers can quickly load pretrained models with the Transformers library and fine-tune "
    "them on custom data to ship production-ready NLP applications. " * 20  # 일부러 길게
)

# BART 기준: 입력 1024 토큰 내 권장. 여유를 두고 900으로 분할(스페셜/헤더 여지)
max_in_tokens = 900
overlap = 50

tok_len = count_tokens(tok_bart, long_text)
print(f"[원문 토큰 수 (BART 토크나이저)]: {tok_len}")

chunks = make_chunks(tok_bart, long_text, max_tokens=max_in_tokens, overlap_tokens=overlap)
print(f"[생성된 청크 개수]: {len(chunks)}")
for ci, c in enumerate(chunks[:2]):  # 앞 2개만 확인
    print(f" - 청크{ci} 토큰수:", count_tokens(tok_bart, c))

# ---------------------------------------------------------------------
# 5) batch_decode/clean_up_tokenization_spaces 옵션 맛보기
#   - 모델 출력 토큰을 다시 텍스트로 돌릴 때 공백/특수문자 정리
# ---------------------------------------------------------------------
p("5) batch_decode / clean_up_tokenization_spaces")

enc_batch = tok_gpt2(batch_texts, padding=True, return_tensors="pt")
# 일반적으로 모델 출력으로부터 ids를 받게 되지만, 여기선 입력 ids로 예시
decoded = tok_gpt2.batch_decode(enc_batch["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True)
print("batch_decode:", decoded)

print("\n[완료] 토크나이저 기본/심화 개념 실습이 끝났습니다.")
