import unittest
from utils.gt.gt import GT
from utils.gt.gt_dummy_detector import GtDummyDetectorMixin


class GtDetector(GtDummyDetectorMixin, GT):
    pass


class GTDetectorTestCase(unittest.TestCase):
    def test_detect(self):
        self.detector = GtDetector(filename='data/GT/Sowbug3_cut.txt',
                                   bb_size_px=70,
                                   fp_prob=0.05, fn_prob=0.001,
                                   false_detection_jitter_scale=40)
        bboxes = self.detector.detect(0)
        for bbox in bboxes:
            self.assertEqual(bbox.width, 70)
            self.assertEqual(bbox.height, 70)

    def test_fn1(self):
        self.detector = GtDetector(filename='data/GT/Sowbug3_cut.txt',
                                   bb_size_px=70,
                                   fp_prob=0, fn_prob=1,
                                   false_detection_jitter_scale=40)
        bboxes = self.detector.detect(0)
        self.assertEqual(len(bboxes), 0)

    def test_no_fp_fn(self):
        self.detector = GtDetector(filename='data/GT/Sowbug3_cut.txt',
                                   bb_size_px=70,
                                   fp_prob=0, fn_prob=0,
                                   false_detection_jitter_scale=40)
        bboxes = self.detector.detect(0)
        self.assertEqual(len(bboxes), 5)

