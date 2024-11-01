import unittest
from evaluate import mask_iou, mask_to_box
import numpy as np


class UtilityTestCase(unittest.TestCase):
    def test_mask_iou(self):
        gt_labels = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ])
        pred_labels = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ])
        self.assertAlmostEqual(mask_iou(pred_labels, gt_labels), 0.5)

    def test_mask_to_box(self):
        mask = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ])
        box = [1, 1, 5, 4]
        self.assertListEqual(mask_to_box(mask).tolist(), box)


if __name__ == '__main__':
    unittest.main()
