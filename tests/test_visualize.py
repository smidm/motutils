import unittest
import mot_utils


class VisualizeTestCase(unittest.TestCase):
    def test_visualize(self):
        gt = mot_utils.Mot('tests/data/Sowbug3_cut.txt')
        mot_utils.visualize('tests/data/Sowbug3_cut.mp4',
                                     'tests/out/Sowbug3_cut_visualize.mp4',
                                     [gt.draw_frame],
                                     duration=1)

    def test_visualize_posemot(self):
        pose_gt = mot_utils.PoseMot(filename='tests/data/Sowbug3_cut_pose.csv')
        gt = mot_utils.Mot(filename='tests/data/Sowbug3_cut.txt')
        mot_utils.visualize('tests/data/Sowbug3_cut.mp4',
                                     'tests/out/Sowbug3_cut_visualize_pose.mp4',
                                     [gt.draw_frame, pose_gt.draw_frame],
                                     duration=1)
