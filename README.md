# hf-ai-backend (Flask 버전, 로컬 pip 설치)

> **핵심 요약 (venv 없이 전역/로컬 파이썬 사용)**

```bash
# pip 최신화 + 의존성 설치
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Hugging Face Transformers 모델을 **빠르게 API로 서빙**하는 백엔드. 텍스트 생성, 감정분석, 번역(영→한), 요약(영/한), OCR(ID) 엔드포인트를 **Flask**로 제공합니다.

---

## 프로젝트 개요

- **스택**: Flask, (선택) gunicorn, Hugging Face Transformers, PyTorch, EasyOCR, Pillow
- **엔드포인트**
  - `GET /health` — 서버/디바이스 상태
  - `POST /generate` — 텍스트 생성 (distilgpt2)
  - `POST /sentiment` — 감정분석
  - `POST /translate/en2ko` — 번역(영→한, NLLB 권장: `facebook/nllb-200-distilled-600M`)
  - `POST /summarize/en`, `POST /summarize/ko` — 요약(영: BART, 한: KoBART)
  - `POST /generate/stream` — 토큰 스트리밍 출력
  - `POST /ocr/id` — **multipart/form-data** `file` 필드 이미지 OCR (EasyOCR/ko+en)
- **디바이스 자동 선택**: `accelerate`가 있으면 `device_map="auto"`로 GPU/MPS 활용, 없으면 CPU 폴백
- **환경변수로 잡음 최소화**:
  - `HF_HUB_ENABLE_HF_TRANSFER=0` (전송가속 비활성)
  - `TOKENIZERS_PARALLELISM=false` (토크나이저 경고 억제)
  - (선택) `HF_HUB_OFFLINE=1` (허브 오프라인 캐시만 사용)

> 주의: Flask에는 `/docs` 자동 문서가 없습니다. cURL/HTTP 클라이언트를 사용하세요.

---

## 1) 의존성 설치 (venv 없이)

### requirements.txt

아래 파일을 프로젝트 루트에 두고 설치하세요.

```txt
flask>=3.0
transformers>=4.42
torch>=2.3
sentencepiece>=0.1.99
# OCR
easyocr>=1.7
pillow>=10.0
numpy>=1.23
```

**옵션(필요 시 개별 설치)**

- `pillow-heif` — HEIC/HEIF 이미지를 열어야 할 때: `python -m pip install pillow-heif`
- `accelerate` — 멀티GPU/자동 디바이스 매핑: `python -m pip install accelerate`
- `gunicorn` — 운영 배포용 WSGI 서버: `python -m pip install gunicorn`

> **PyTorch 주의**
>
> - 일반적으로 위 명령으로 CPU/MPS에 맞는 torch가 설치됩니다.
> - CPU 전용 휠을 강제하려면:\
>   `python -m pip install "torch>=2.3" --index-url https://download.pytorch.org/whl/cpu`

---

## 2) 서버 실행

```bash
# 개발 서버
python -m app.main     # 또는: python app/main.py

# (선택) 배포 서버 예시 (gunicorn)
# pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:8000 app.main:app
```

헬스체크:

```bash
curl -s http://127.0.0.1:8000/health | jq
```

---

## 3) 빠른 호출 예시 (cURL)

```bash
# 텍스트 생성
curl -s -X POST 'http://127.0.0.1:8000/generate' \
  -H 'Content-Type: application/json' \
  --data-binary '{"prompt":"Once upon a time","max_new_tokens":40,"temperature":0.8}' | jq

# 감정분석
curl -s -X POST 'http://127.0.0.1:8000/sentiment' \
  -H 'Content-Type: application/json' \
  --data-binary '{"text":"I love Hugging Face!"}' | jq

# 번역(영→한) — NLLB 권장
curl -s -X POST 'http://127.0.0.1:8000/translate/en2ko' \
  -H 'Content-Type: application/json' \
  --data-binary '{"texts":["How are you?","This API works great."],"num_beams":4}' | jq

# 요약(영)
curl -s -X POST 'http://127.0.0.1:8000/summarize/en' \
  -H 'Content-Type: application/json' \
  --data-binary '{"text":"Hugging Face provides an open platform...", "max_length":60, "min_length":20, "num_beams":4, "no_repeat_ngram_size":3}' | jq

# 요약(한)
curl -s -X POST 'http://127.0.0.1:8000/summarize/ko' \
  -H 'Content-Type: application/json' \
  --data-binary '{"text":"허깅페이스는 다양한 머신러닝 모델과 ...", "max_length":80, "min_length":30, "num_beams":4, "no_repeat_ngram_size":3}' | jq

# 스트리밍 생성 (서버가 토큰을 즉시 흘려보냄)
curl -N -s -X POST 'http://127.0.0.1:8000/generate/stream' \
  -H 'Content-Type: application/json' \
  --data-binary '{"prompt":"Hello AI","max_new_tokens":60,"temperature":0.8}'

# OCR(ID) — multipart/form-data (필드명: file)
curl -v -F "file=@./id_sample.jpeg" \
  "http://127.0.0.1:8000/ocr/id?parse=true"

# (선택) 크롭 적용
curl -v -F "file=@./id_sample.jpeg" \
  "http://127.0.0.1:8000/ocr/id?parse=true&crop=120,200,800,400"
```

---

## 4) 환경 변수(권장)

```bash
# 네트워크 전송가속/토크나이저 경고 억제
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false

# (선택) 허브 오프라인 캐시만 사용
# export HF_HUB_OFFLINE=1

# (선택) OCR 입력 최대 크기 변경 (기본 5MB)
# export OCR_MAX_IMAGE_BYTES=$((10*1024*1024))
```

---

## 5) 트러블슈팅

| 증상                                                   | 원인                  | 해결                                                      |
| ---------------------------------------------------- | ------------------- | ------------------------------------------------------- |
| `Using device_map requires accelerate`               | `accelerate` 미설치    | `python -m pip install -U accelerate` 또는 CPU로 실행        |
| `MarianTokenizer requires the SentencePiece library` | `sentencepiece` 미설치 | `python -m pip install -U sentencepiece`                |
| `/ocr/id` 400에 `file is required`                    | 멀티파트 필드 누락/경계 오류    | `curl -F "file=@path.jpg" http://.../ocr/id` 형태로 테스트    |
| `/ocr/id` 400에 `이미지 파일을 열 수 없습니다.`                   | HEIC 등 미지원 포맷       | `python -m pip install pillow-heif` 후 재시작, JPG/PNG로 전송  |
| 번역 품질 불안정                                            | 언어코드/빔 미설정          | NLLB에서 `num_beams=4`, `src_lang/tgt_lang` 명시(코드는 이미 설정) |
| 한글 요약 토큰 초과/짤림                                       | 입력 장문               | 길이 줄이거나 모델 `model_max_length=1024` 설정(코드 반영됨)           |

---

## 6) 구조(요약)

```
app/
├── main.py           # Flask 앱 생성/미들웨어/블루프린트 등록
├── middleware.py     # Request-ID, 로깅, 에러 JSON 핸들러
├── routes.py         # 모든 엔드포인트(텍스트, 요약/번역, 스트림, OCR)
├── pipelines.py      # HF 파이프라인 및 EasyOCR 로더(게으른 로딩)
├── cache.py          # 초간단 LRU 응답 캐시
├── config.py         # 환경/디바이스/경고 설정, 오프라인 모드, MPS 캐시 정리
├── utils.py          # 입력 가드/이미지 바이트 가드/crop 파서
└── __init__.py       # Flask app 재노출
```
