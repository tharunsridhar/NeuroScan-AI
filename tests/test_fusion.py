import numpy as np

from core.classifier_fusion import adaptive_model_fusion


def test_adaptive_model_fusion_returns_consistent_prediction():
    raw_probs = {
        "EfficientNetV2-S": np.array([0.8, 0.1, 0.05, 0.05], dtype=np.float32),
        "MobileNetV3": np.array([0.7, 0.15, 0.1, 0.05], dtype=np.float32),
        "ConvNeXt Tiny": np.array([0.75, 0.1, 0.1, 0.05], dtype=np.float32),
    }
    result = adaptive_model_fusion(raw_probs, 0.9)
    assert result["final_class"] == "glioma"
    assert result["fused_confidence"] > 0.5
