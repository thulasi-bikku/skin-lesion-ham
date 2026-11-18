"""Model zoo: pretrained backbones and two from-scratch baselines."""
import torch.nn as nn
from torchvision import models

from .common import NUM_CLASSES

INPUT_SIZE = {"inceptionv3": 299, "mobilenetv2": 224, "efficientnet_b0": 224, "efficientnet_v2_s": 224,
              "convnext_tiny": 224, "swin_t": 224, "vit_small": 224, "dinov2_small": 224,
              "simple_cnn": 224, "cnn_lstm": 224}
PRETRAINED = {"inceptionv3", "mobilenetv2", "efficientnet_b0", "efficientnet_v2_s", "convnext_tiny",
              "swin_t", "vit_small", "dinov2_small"}


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    """Four-stage convolutional network trained from scratch."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(_block(3, 32), _block(32, 64), _block(64, 128), _block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.4),
                                  nn.Linear(256, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


class CNNLSTM(nn.Module):
    """Convolutional stem whose feature-map rows are read as a sequence by an LSTM."""

    def __init__(self, num_classes=NUM_CLASSES, hidden=256):
        super().__init__()
        self.features = nn.Sequential(_block(3, 32), _block(32, 64), _block(64, 128))
        self.pool = nn.AdaptiveAvgPool2d((14, 14))
        self.lstm = nn.LSTM(input_size=128 * 14, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(hidden, num_classes))

    def forward(self, x):
        f = self.pool(self.features(x))                    # B, C, H, W
        b, c, h, w = f.shape
        seq = f.permute(0, 2, 1, 3).reshape(b, h, c * w)   # B, H, C*W
        out, _ = self.lstm(seq)
        return self.head(out[:, -1])


def build_model(name, pretrained=True, num_classes=NUM_CLASSES):
    """Return (model, head_module); head_module is trained first when fine-tuning."""
    if name == "inceptionv3":
        w = models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.inception_v3(weights=w, aux_logits=True, transform_input=True, init_weights=not pretrained)
        m.aux_logits, m.AuxLogits = False, None
        m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, num_classes))
        return m, m.fc
    if name == "mobilenetv2":
        w = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.mobilenet_v2(weights=w)
        m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.last_channel, num_classes))
        return m, m.classifier
    if name == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.classifier[1].in_features, num_classes))
        return m, m.classifier
    if name == "efficientnet_v2_s":
        w = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_v2_s(weights=w)
        m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.classifier[1].in_features, num_classes))
        return m, m.classifier
    if name == "convnext_tiny":
        w = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.convnext_tiny(weights=w)
        m.classifier[2] = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.classifier[2].in_features, num_classes))
        return m, m.classifier
    if name == "swin_t":
        w = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.swin_t(weights=w)
        m.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.head.in_features, num_classes))
        return m, m.head
    if name == "vit_small":
        import timm
        m = timm.create_model("vit_small_patch16_224", pretrained=pretrained, num_classes=num_classes,
                              drop_rate=0.4)
        return m, m.head
    if name == "dinov2_small":
        import timm
        m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=pretrained,
                              num_classes=num_classes, img_size=224, drop_rate=0.4)
        return m, m.head
    if name == "simple_cnn":
        m = SimpleCNN(num_classes)
        return m, m.head
    if name == "cnn_lstm":
        m = CNNLSTM(num_classes)
        return m, m.head
    raise ValueError(f"Unknown model: {name}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
