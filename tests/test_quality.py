import numpy as np

from core.quality_metrics import quality_metrics


def test_quality_metrics_returns_expected_keys():
    result = quality_metrics(np.zeros((64, 64, 3), dtype=np.uint8))
    assert {"quality_score", "blur_metric", "brightness_metric", "contrast_metric", "usable_for_analysis"} <= set(result)
