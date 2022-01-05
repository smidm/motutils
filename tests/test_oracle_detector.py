import unittest

import motutils


class OracleDetector(motutils.OracleDetectorMixin, motutils.Mot):
    pass


class GTDetectorTestCase(unittest.TestCase):
    def test_detect(self):
        self.detector = OracleDetector(
            filename_or_buffer="tests/data/Sowbug3_cut.csv",
            bb_size_px=70,
            fp_prob=0.05,
            fn_prob=0.001,
            false_detection_jitter_scale=40,
        )
        bboxes = self.detector.detect(0)
        for bbox in bboxes:
            self.assertAlmostEqual(bbox.width, 70)
            self.assertAlmostEqual(bbox.height, 70)
            # on py39 on travis this:
            #    self.assertEqual(bbox.width, 70)
            # fails with:
            # AssertionError: 70.00000000000003 != 70
            # why?
            # py36-38 are ok

    def test_fn1(self):
        self.detector = OracleDetector(
            filename_or_buffer="tests/data/Sowbug3_cut.csv",
            bb_size_px=70,
            fp_prob=0,
            fn_prob=1,
            false_detection_jitter_scale=40,
        )
        bboxes = self.detector.detect(0)
        self.assertEqual(len(bboxes), 0)

    def test_no_fp_fn(self):
        self.detector = OracleDetector(
            filename_or_buffer="tests/data/Sowbug3_cut.csv",
            bb_size_px=70,
            fp_prob=0,
            fn_prob=0,
            false_detection_jitter_scale=40,
        )
        bboxes = self.detector.detect(0)
        self.assertEqual(len(bboxes), 5)
