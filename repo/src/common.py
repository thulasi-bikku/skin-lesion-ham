"""Shared constants for the HAM10000 skin-lesion classification pipeline."""

CLASS_CODES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_NAMES = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}
NUM_CLASSES = len(CLASS_CODES)
CODE_TO_IDX = {c: i for i, c in enumerate(CLASS_CODES)}

# Models evaluated in the study, in the order used in tables and figures.
MODEL_ORDER = ["simple_cnn", "cnn_lstm", "mobilenetv2", "inceptionv3", "efficientnet_b0",
               "efficientnet_v2_s", "convnext_tiny", "swin_t", "vit_small", "dinov2_small"]
MODEL_LABELS = {
    "simple_cnn": "Baseline CNN",
    "cnn_lstm": "CNN-LSTM",
    "mobilenetv2": "MobileNetV2",
    "inceptionv3": "InceptionV3",
    "efficientnet_b0": "EfficientNet-B0",
    "vit_small": "ViT-S/16",
    "efficientnet_v2_s": "EfficientNetV2-S",
    "convnext_tiny": "ConvNeXt-T",
    "swin_t": "Swin-T",
    "dinov2_small": "DINOv2 ViT-S/14",
}
