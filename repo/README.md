# Skin lesion classification on HAM10000 with a lesion-level evaluation protocol

Code and manuscript source for the paper *An artificial intelligence-assisted decision support system for skin
lesion classification from dermoscopic images* (T. Bikku and J. J. Pujari).

The repository trains and evaluates ten classifiers (InceptionV3, MobileNetV2, EfficientNet-B0, EfficientNetV2-S,
ConvNeXt-T, Swin-T, ViT-S/16, DINOv2 ViT-S/14, a convolutional baseline and a CNN-LSTM) on the seven diagnostic classes of HAM10000. Every number, table and figure in
the paper is produced by this code; nothing is entered by hand.

## Evaluation protocol

* Lesion-level, stratified split (about 70/10/20 for train/validation/test): all images of one lesion stay in one partition.
* Class-weighted cross-entropy, AdamW, two-stage fine-tuning, checkpoint selection on validation macro-F1.
* Three training seeds per model; reported as mean and standard deviation.
* Accuracy, balanced accuracy, macro-F1, MCC, AUROC, per-class sensitivity and specificity, 95% bootstrap intervals,
  and the accuracy of a majority-class predictor as a reference.
* Grad-CAM saliency maps for InceptionV3.

## Reported results

Single training run (seed 0) per model on a Tesla T4; test partition of 2,020 images (majority-class accuracy 0.661).
The raw outputs are in `results/` (metrics, training histories, test predictions, partition).

| Model | Accuracy | Balanced acc. | Macro-F1 | MCC | AUROC |
|---|---|---|---|---|---|
| Baseline CNN | 0.724 | 0.389 | 0.398 | 0.472 | 0.891 |
| MobileNetV2 | 0.835 | 0.731 | 0.723 | 0.684 | 0.969 |
| InceptionV3 | 0.798 | 0.694 | 0.610 | 0.631 | 0.948 |
| ConvNeXt-T | 0.850 | 0.739 | 0.767 | 0.707 | 0.971 |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download HAM10000 (Kaggle `kmader/skin-cancer-mnist-ham10000` or the ISIC archive) and place
`HAM10000_metadata.csv` and all `.jpg` images below one folder, for example `data/HAM10000/`.
The images may be spread over sub-folders.

## Reproduce the paper

```bash
bash scripts/run_all.sh data/HAM10000
```

This trains all models, writes `outputs/<model>/seed<k>/` (metrics, history, predictions, checkpoint), builds all
figures (EPS, PDF and 600 dpi PNG) and LaTeX tables in `paper/figures` and `paper/generated`, and compiles
`paper/main.pdf`. A GPU is strongly recommended; set `MODELS="inceptionv3 mobilenetv2"` to train only a subset. Individual steps:

```bash
python -m src.train --data-dir data/HAM10000 --model inceptionv3 --seeds 0 1 2
python -m src.diagrams
python -m src.report --out-dir outputs --paper-dir paper --data-dir data/HAM10000 --primary inceptionv3
python -m src.gradcam --data-dir data/HAM10000 --model inceptionv3 --checkpoint outputs/inceptionv3/seed0/best.pt
```

## Word version

After the run, `bash paper/build.sh && bash paper/to_docx.sh` writes `paper/manuscript.docx` (requires `pandoc`). Figure and table numbers and cross-references are resolved from the LaTeX build.

## Tests

`tests/test_report_pipeline.py` checks the lesion-grouped split, the metrics and the report generator on synthetic
data (no images or GPU needed): `python -m pytest tests`.

## Repository layout

```
src/        data, models, training, evaluation, report and Grad-CAM code
scripts/    run_all.sh
paper/      main.tex (JAMI format), build.sh, generated/ and figures/ (written by the pipeline)
tests/      CPU-only tests
results/    raw outputs of the reported runs (no checkpoints)
notebooks/  kaggle_run.ipynb: run the experiments on a free Kaggle GPU
```

## Notes

* The manuscript follows the JAMI style: 12 pt, double spacing, 12.2 cm x 19.3 cm text area.
* HAM10000 is not redistributed here; follow the licence of the original dataset.
* This is research code. It is not a medical device and must not be used for clinical decisions.

## Licence

Code: MIT (see `LICENSE`).
