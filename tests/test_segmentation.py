from pathlib import Path

import cv2
import numpy as np

from core.segmentation import run_segmentation


class DummyTensor:
    def __init__(self, array):
        self._array = array

    def numpy(self):
        return self._array


class DummySegmentationModel:
    input_shape = (None, 256, 256, 1)

    def __call__(self, inputs, training=False):
        return DummyTensor(np.ones((1, 256, 256, 1), dtype=np.float32) * 0.8)


def test_run_segmentation_outputs_binary_mask(tmp_path: Path):
    image_path = tmp_path / "scan.png"
    cv2.imwrite(str(image_path), np.zeros((256, 256, 3), dtype=np.uint8))
    mask = run_segmentation(DummySegmentationModel(), str(image_path))
    assert mask.shape == (384, 384)
    assert set(np.unique(mask)).issubset({0, 1})
