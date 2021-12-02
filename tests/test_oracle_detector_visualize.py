import unittest
from utils.gt.mot import Mot
from utils.gt.oracle_detector import OracleDetectorMixin
from utils.gt.mot_project import MotProjectMixin
# import numpy as np
import os


class OracleDetector(MotProjectMixin, OracleDetectorMixin, Mot):
    pass


class OracleDetectorVisualizeTestCase(unittest.TestCase):
    def setUp(self):
        self.detector = OracleDetector(filename='data/GT/Sowbug3_cut.txt',
                                       bb_size_px=70,
                                       fp_prob=0.05, fn_prob=0.001,
                                       false_detection_jitter_scale=40)

    def test_visualize(self):
        import cv2
        from core.project.project import Project
        p = Project('test/project/Sowbug3_cut_300_frames')
        self.detector.set_project_offsets(p)

        try:
            os.makedirs('test/out/OracleDetectorVisualizeTestCase')
        except OSError:
            pass

        for frame in range(10):
            img = p.img_manager.get_whole_img(frame)
            for bbox in self.detector.detect(frame):
                bbox.draw_to_image(img)
                # bbox.move(-np.array([p.video_crop_model['x1'], p.video_crop_model['y1']])).draw_to_image(img)
            cv2.imwrite('test/out/OracleDetectorVisualizeTestCase/{:03d}.png'.format(frame), img)

