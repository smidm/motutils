import unittest

import motutils


class OracleDetector(motutils.OracleDetectorMixin, motutils.Mot):
    pass


class OracleDetectorVisualizeTestCase(unittest.TestCase):
    def setUp(self):
        self.detector = OracleDetector(
            filename_or_buffer="tests/data/Sowbug3_cut.csv",
            bb_size_px=70,
            fp_prob=0.05,
            fn_prob=0.001,
            false_detection_jitter_scale=40,
        )

    def test_visualize(self):
        import cv2

        cap = cv2.VideoCapture("tests/data/Sowbug3_cut.mp4")
        for frame in range(10):
            ret, img = cap.read()
            assert ret
            for bbox in self.detector.detect(frame):
                bbox.draw_to_image(img)
                # bbox.move(-np.array([p.video_crop_model['x1'], p.video_crop_model['y1']])).draw_to_image(img)
            cv2.imwrite(
                "tests/out/OracleDetectorVisualizeTestCase_{:03d}.png".format(frame),
                img,
            )
