# hf-ai-backend

> 바로 설치(요약)
```bash
# (선택) venv 생성 및 활성화
python3 -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# pip 업그레이드 + 의존성 설치
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

허깅페이스(Transformers) 모델을 **빠르게 API로 서빙**하는 백엔드 스타터입니다.  
텍스트 생성, 감정분석, 번역(영→한), 요약(영/한) 엔드포인트를 FastAPI로 제공합니다.

---

## 프로젝트 설명

- **스택**: FastAPI, Uvicorn, Hugging Face Transformers, PyTorch  
- **엔드포인트**
  - `GET /health` — 서버/디바이스 상태
  - `POST /generate` — 텍스트 생성 (distilgpt2)
  - `POST /sentiment` — 감정분석
  - `POST /translate/en2ko` — 번역(영→한, NLLB 권장)
  - `POST /summarize/en`, `POST /summarize/ko` — 요약(영/BART, 한/KoBART)
- **디바이스 자동 선택**: `accelerate`가 설치되어 있으면 GPU/MPS 자동 선택, 없으면 CPU로 폴백  
- **네트워크/토크나이저 잡음 최소화**: 환경변수로 전송가속/병렬경고를 끔

---

## 1) 가상환경(venv) 셋업

> 팀 저장소에는 `venv/`를 **커밋하지 않습니다**. 각자 로컬에서 생성하세요.

```bash
# 1) 가상환경 만들기 (폴더명은 .venv 권장)
python3 -m venv .venv

# 2) 활성화
source .venv/bin/activate

# 3) 현재 파이썬/환경 확인
which python
python -V
```

> venv를 쓰지 않고 **전역**에서 진행해도 됩니다. 아래 pip 설치만 동일하게 하면 됩니다.

---

## 2) pip 셋팅 & 의존성 설치

### (권장) requirements.txt 최소 세트
```txt
fastapi[all]>=0.110
uvicorn[standard]>=0.23
transformers>=4.41
huggingface_hub>=0.23
accelerate>=0.26        # device_map="auto"에 필요 (없으면 CPU로 폴백)
torch>=2.3              # PyTorch (Apple Silicon은 기본 인덱스로 OK)
sentencepiece>=0.1.99   # Marian/NLLB 번역 모델에 필요
```

설치 명령:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **PyTorch 주의**  
> - Apple Silicon(M1/M2): 위 명령 그대로 OK  
> - Intel/일반 CPU에서 속도보다 호환 우선이면:  
>   `python -m pip install "torch>=2.3" --index-url https://download.pytorch.org/whl/cpu`

> **번역 에러(ImportError: sentencepiece)** 나면  
> `python -m pip install -U sentencepiece`

---

## 3) 서버 실행

프로젝트에 포함된 FastAPI 앱 파일을 실행하세요. (예: `app/api.py`)

```bash
# 서버 실행 (둘 중 하나)
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

확인:
```bash
# 헬스체크
curl -s http://127.0.0.1:8000/health | jq

# 자동 문서
# 브라우저에서: http://127.0.0.1:8000/docs
```

---

## 4) 빠른 호출 예시 (cURL)

```bash
# 텍스트 생성
curl -s -X POST 'http://127.0.0.1:8000/generate'   -H 'Content-Type: application/json'   --data-binary '{"prompt":"Once upon a time","max_new_tokens":40,"temperature":0.8}' | jq

# 감정분석
curl -s -X POST 'http://127.0.0.1:8000/sentiment'   -H 'Content-Type: application/json'   --data-binary '{"text":"I love Hugging Face!"}' | jq

# 번역(영→한) — NLLB 권장 (서버에 NLLB 설정 가정)
curl -s -X POST 'http://127.0.0.1:8000/translate/en2ko'   -H 'Content-Type: application/json'   --data-binary '{"texts":["How are you?","This API works great."],"num_beams":4}' | jq

# 요약(영)
curl -s -X POST 'http://127.0.0.1:8000/summarize/en'   -H 'Content-Type: application/json'   --data-binary '{"text":"Hugging Face provides an open platform...", "max_length":60, "min_length":20, "num_beams":4, "no_repeat_ngram_size":3}' | jq

# 요약(한)
curl -s -X POST 'http://127.0.0.1:8000/summarize/ko'   -H 'Content-Type: application/json'   --data-binary '{"text":"허깅페이스는 다양한 머신러닝 모델과 ...", "max_length":80, "min_length":30, "num_beams":4, "no_repeat_ngram_size":3}' | jq
```

---

## 트러블슈팅 빠른 표

| 증상 | 원인 | 해결 |
|---|---|---|
| `Using device_map requires accelerate` | accelerate 미설치 | `python -m pip install -U accelerate` 혹은 코드에서 CPU 폴백 |
| `MarianTokenizer requires the SentencePiece library` | sentencepiece 미설치 | `python -m pip install -U sentencepiece` |
| `hf_transfer dns error` 경고 | 전송가속 네트워크 이슈 | `export HF_HUB_ENABLE_HF_TRANSFER=0` |
| `tokenizers parallelism` 경고 | fork 후 병렬처리 경고 | `export TOKENIZERS_PARALLELISM=false` |
| 번역이 엉뚱한 결과 | 언어쌍/언어코드 미명시 | task를 `translation_en_to_ko`로, 또는 NLLB `src_lang/tgt_lang` 명시 |
