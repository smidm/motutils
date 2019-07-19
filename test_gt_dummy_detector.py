import unittest
from utils.gt.gt import GT
from utils.gt.gt_dummy_detector import GtDummyDetectorMixin


class GtDetector(GtDummyDetectorMixin, GT):
    pass


class GTDetectorTestCase(unittest.TestCase):
    def setUp(self):
        self.detector = GtDetector('data/GT/Sowbug3_cut.txt')
        self.detector.bbox_size_px = 70

    def test_detect(self):
        self.detector.init_gt_dummy_detector(self.detector.bbox_size_px,
                                             fp_prob=0.05, fn_prob=0.001,
                                             false_detection_jitter_scale=40)
        bboxes = self.detector.detect(0)
        for bbox in bboxes:
            self.assertEqual(bbox.width, self.detector.bbox_size_px)
            self.assertEqual(bbox.height, self.detector.bbox_size_px)

    def test_fn1(self):
        self.detector.init_gt_dummy_detector(self.detector.bbox_size_px,
                                             fp_prob=0, fn_prob=1,
                                             false_detection_jitter_scale=40)
        bboxes = self.detector.detect(0)
        self.assertEqual(len(bboxes), 0)

    def test_no_fp_fn(self):
        self.detector.init_gt_dummy_detector(self.detector.bbox_size_px,
                                             fp_prob=0, fn_prob=0,
                                             false_detection_jitter_scale=40)
        bboxes = self.detector.detect(0)
        self.assertEqual(len(bboxes), 5)

