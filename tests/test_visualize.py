import unittest
import mot_utils


class VisualizeTestCase(unittest.TestCase):
    def test_visualize(self):
        gt = mot_utils.Mot('data/GT/Sowbug3_cut.txt')
        mot_utils.visualize('/datagrid/ferda/data/youtube/Sowbug3_cut.mp4',
                                     'test/out/gt_visualize1.mp4',
                                     [gt.draw_frame],
                                     duration=1)

    def test_visualize_posemot(self):
        pose_gt = mot_utils.PoseMot(filename='data/GT/Cam1_clip.avi_pose.csv')
        gt = mot_utils.Mot(filename='data/GT/Cam1_clip.avi.txt')
        mot_utils.visualize('/datagrid/ferda/data/ants_ist/camera_1/Cam1_clip.avi',
                                     'test/out/gt_visualize2.mp4',
                                     [gt.draw_frame, pose_gt.draw_frame],
                                     duration=1)
