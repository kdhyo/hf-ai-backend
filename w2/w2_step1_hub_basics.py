# w2_step1_hub_basics.py
# ------------------------------------------------------------
# Hub 리포 구조 파악 + 안전한 스냅샷 다운로드(가속전송 OFF, 불필요 파일 제외)
# ------------------------------------------------------------
import os

# ✅ 가속 전송(transfer.xethub.hf.co) 비활성화 → 사내/특정 네트워크에서 멈춤 방지
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import list_repo_files, hf_hub_download, snapshot_download
from transformers import AutoConfig, AutoTokenizer

MODEL_ID = "distilgpt2"

# 1) 리포 파일 나열
files = list_repo_files(MODEL_ID)
print("[리포 파일 수]:", len(files))
print("[상위 10개]:", files[:10])

# 2) README 일부 확인
readme_path = hf_hub_download(repo_id=MODEL_ID, filename="README.md")
print("[README.md 경로]:", readme_path)
print("--- README 헤더 ---")
with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f):
        if i > 9: break
        print(line.rstrip())

# 3) config/tokenizer 메타
cfg = AutoConfig.from_pretrained(MODEL_ID)
print("[모델 아키텍처]:", cfg.architectures)
print("[vocab_size/ n_layer/ n_head]:", getattr(cfg, "vocab_size", None), getattr(cfg, "n_layer", None), getattr(cfg, "n_head", None))

tok = AutoTokenizer.from_pretrained(MODEL_ID)
print("[pad_token]:", tok.pad_token, "/ [eos_token]:", tok.eos_token)
print("[모델 최대 길이(model_max_length)]:", tok.model_max_length)

# 4) 스냅샷 다운로드 (필요 파일만, 재시도/재개)
try:
    local_dir = snapshot_download(
        repo_id=MODEL_ID,
        # ✅ 꼭 필요한 것만 받기 (대형 coreml/tflite/onnx/msgpack 제외)
        allow_patterns=[
            "config.json",
            "tokenizer.json", "vocab.json", "merges.txt",
            "pytorch_model.bin",
            "generation_config.json",
        ],
        ignore_patterns=[
            "*.tflite", "*.onnx", "*.mlmodel", "coreml/*", "*.msgpack", "*.safetensors"
        ],
        resume_download=True,   # 중간부터 재개
        max_workers=2,          # 과한 병렬 ↓ (안정성↑)
        etag_timeout=10,        # 메타 요청 타임아웃
    )
    print("[스냅샷 디렉토리]:", local_dir)

except Exception as e:
    # 네트워크/프록시 이슈 시 개별 파일 폴백
    print("[경고] snapshot_download 실패, 개별 파일로 폴백:", e)
    needed = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
    ]
    paths = []
    for fn in needed:
        try:
            p = hf_hub_download(repo_id=MODEL_ID, filename=fn)
            print(" - ok:", fn)
            paths.append(p)
        except Exception as e2:
            print(f" - skip {fn} ({e2})")
    if paths:
        print("[개별 다운로드 완료, 예시 경로]:", paths[0])
    else:
        print("[실패] 네트워크/권한 문제로 필수 파일 다운로드 불가")
