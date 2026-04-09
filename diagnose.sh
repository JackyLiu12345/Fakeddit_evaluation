#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

SPLIT="${1:-test}"
TASK="${2:-2}"
SAMPLE_FRACTION="${3:-0.01}"
SMOKE="${SMOKE:-0}"
SMOKE_OUTPUT="${REPO_ROOT}/results_diagnose_smoke.csv"

echo "== [1/8] Environment sanity =="
python - <<'PY'
import platform
import torch
print("python:", platform.python_version())
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
PY

echo
echo "== [2/8] Config snapshot =="
python - <<'PY'
import config as c
print("DEFAULT_MODEL:", c.DEFAULT_MODEL)
print("DEFAULT_SAMPLE_FRACTION:", c.DEFAULT_SAMPLE_FRACTION)
print("DEFAULT_DEVICE:", c.DEFAULT_DEVICE)
print("GENERATION_CONFIG:", c.GENERATION_CONFIG)
print("LABEL_COLUMN:", c.LABEL_COLUMN)
for t in (2, 3, 6):
    print(f"LABEL_MAPS[{t}]:", c.LABEL_MAPS[t])
PY

echo
echo "== [3/8] CLI sanity =="
python evaluate.py --help >/dev/null
python metrics.py --help >/dev/null
echo "evaluate.py and metrics.py CLI parse OK"

echo
echo "== [4/8] Data loading + multimodal filter (split=${SPLIT}, sample_fraction=${SAMPLE_FRACTION}) =="
python - "$SPLIT" "$SAMPLE_FRACTION" <<'PY'
import sys
from data_loader import load_split
split = sys.argv[1]
frac = float(sys.argv[2])
df = load_split(split, sample_fraction=frac)
print("rows:", len(df))
print("columns:", list(df.columns))
if len(df):
    has_image_ok = (
        df["hasImage"].isin([True, 1])
        | df["hasImage"].astype(str).str.lower().isin({"true", "1"})
    )
    has_url_ok = df["image_url"].notna() & (df["image_url"].str.strip() != "")
    print("hasImage true/1 ratio:", float(has_image_ok.mean()))
    print("non-empty image_url ratio:", float(has_url_ok.mean()))
    print("sample ids:", df["id"].head(5).tolist() if "id" in df.columns else [])
PY

echo
echo "== [5/8] Label distribution check (task=${TASK}) =="
python - "$SPLIT" "$SAMPLE_FRACTION" "$TASK" <<'PY'
import sys
import config as cfg
from data_loader import load_split
split = sys.argv[1]
frac = float(sys.argv[2])
task = int(sys.argv[3])
df = load_split(split, sample_fraction=frac)
label_col = cfg.LABEL_COLUMN[task]
print("label_col:", label_col)
print(df[label_col].map(cfg.LABEL_MAPS[task]).value_counts(dropna=False))
PY

echo
echo "== [6/8] Label parser behavior =="
python - <<'PY'
from evaluate import _parse_predicted_label
samples = [
    "real", "fake", "This looks fake.",
    "Probably real news.",
    "true", "fake_with_true_text", "fake_with_false_text",
    "I cannot determine."
]
for s in samples:
    print(f"{s!r} -> task2:{_parse_predicted_label(s, 2)} task3:{_parse_predicted_label(s, 3)}")
PY

echo
echo "== [7/8] Image downloader spot-check =="
python - <<'PY'
import functools
import tempfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image
from utils import download_image

_TEST_IMAGE_WIDTH = 32
_TEST_IMAGE_HEIGHT = 24
_TEST_IMAGE_RGB = (12, 34, 56)
_SERVER_JOIN_TIMEOUT_SEC = 2


class _SilentHandler(SimpleHTTPRequestHandler):
    def log_message(self, _message_format, *args):
        pass


with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = Path(tmpdir)
    Image.new(
        "RGB",
        (_TEST_IMAGE_WIDTH, _TEST_IMAGE_HEIGHT),
        color=_TEST_IMAGE_RGB,
    ).save(tmp_path / "ok.png")

    handler = functools.partial(_SilentHandler, directory=tmpdir)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        checks = [
            (f"http://{host}:{port}/ok.png", True),
            (f"http://{host}:{port}/missing.png", False),
        ]
        for url, should_succeed in checks:
            try:
                img = download_image(url)
                if should_succeed:
                    print("OK:", url, "size=", img.size, "mode=", img.mode)
                else:
                    print("UNEXPECTED_OK:", url, "size=", img.size, "mode=", img.mode)
            except Exception as e:
                if should_succeed:
                    print("FAIL:", url, "->", e)
                else:
                    print("EXPECTED_FAIL:", url, "->", e)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=_SERVER_JOIN_TIMEOUT_SEC)
        if thread.is_alive():
            print(
                "WARN: local test server thread did not stop within",
                _SERVER_JOIN_TIMEOUT_SEC,
                "seconds.",
            )
PY

echo
echo "== [8/8] Optional smoke evaluation (SMOKE=1 to enable) =="
if [[ "$SMOKE" == "1" ]]; then
  python evaluate.py \
    --split "$SPLIT" \
    --task "$TASK" \
    --sample-fraction "$SAMPLE_FRACTION" \
    --output "$SMOKE_OUTPUT" \
    --verbose

  python - "$SMOKE_OUTPUT" "$TASK" <<'PY'
import sys
import pandas as pd
out = sys.argv[1]
task = int(sys.argv[2])
df = pd.read_csv(out)
print("smoke_output:", out)
print("rows:", len(df))
print("predicted_label counts:")
print(df["predicted_label"].value_counts(dropna=False))
print("IMAGE_DOWNLOAD_ERROR:", int(df["raw_response"].astype(str).str.startswith("IMAGE_DOWNLOAD_ERROR").sum()))
print("INFERENCE_ERROR:", int(df["raw_response"].astype(str).str.startswith("INFERENCE_ERROR").sum()))
print("unknown predictions:", int((df["predicted_label"] == "unknown").sum()))
print("\nRecomputed metrics:")
import subprocess
subprocess.run(["python", "metrics.py", out, "--task", str(task)], check=True)
PY
else
  echo "Skipped. Run with: SMOKE=1 ./diagnose.sh [split] [task] [sample_fraction]"
fi

echo
echo "Diagnostics complete."
