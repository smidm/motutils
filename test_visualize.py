import unittest
import utils.gt.mot
import utils.gt.posemot
import utils.gt.visualize


class VisualizeTestCase(unittest.TestCase):
    def test_visualize(self):
        gt = utils.gt.mot.Mot('data/GT/Sowbug3_cut.txt')
        utils.gt.visualize.visualize('/datagrid/ferda/data/youtube/Sowbug3_cut.mp4',
                                     'test/out/gt_visualize1.mp4',
                                     [gt],
                                     duration=1)

    def test_visualize_posemot(self):
        pose_gt = utils.gt.posemot.PoseMot(filename='data/GT/Cam1_clip.avi_pose.csv')
        gt = utils.gt.mot.Mot(filename='data/GT/Cam1_clip.avi.txt')
        utils.gt.visualize.visualize('/datagrid/ferda/data/ants_ist/camera_1/Cam1_clip.avi',
                                     'test/out/gt_visualize2.mp4',
                                     [gt, pose_gt],
                                     duration=1)
