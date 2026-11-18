#!/usr/bin/env bash
# Full reproduction: train all models (3 seeds each), then build figures, tables and the PDF.
# Usage: bash scripts/run_all.sh /path/to/HAM10000
set -euo pipefail
DATA_DIR="${1:?Usage: bash scripts/run_all.sh /path/to/HAM10000}"
OUT_DIR="${OUT_DIR:-outputs}"

MODELS="${MODELS:-simple_cnn cnn_lstm mobilenetv2 inceptionv3 efficientnet_b0 efficientnet_v2_s convnext_tiny swin_t vit_small dinov2_small}"
for MODEL in $MODELS; do
  python -m src.train --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" --model "$MODEL" --seeds 0 1 2
done

python -m src.diagrams
python -m src.report --out-dir "$OUT_DIR" --paper-dir paper --data-dir "$DATA_DIR" --primary inceptionv3
python -m src.gradcam --data-dir "$DATA_DIR" --splits "$OUT_DIR/splits.csv" --model inceptionv3 \
  --checkpoint "$OUT_DIR/inceptionv3/seed0/best.pt" --out paper/figures/fig_gradcam

bash paper/build.sh
