import unittest
import utils.gt.gt
import utils.gt.posegt
import utils.gt.visualize


class VisualizeTestCase(unittest.TestCase):
    def test_visualize(self):
        gt = utils.gt.gt.GT('data/GT/Sowbug3_cut.txt')
        utils.gt.visualize.visualize('/datagrid/ferda/data/youtube/Sowbug3_cut.mp4',
                                     'test/out/gt_visualize1.mp4',
                                     [gt],
                                     duration=1)

    def test_visualize_posegt(self):
        pose_gt = utils.gt.posegt.PoseGt(filename='data/GT/Cam1_clip.avi_pose.csv')
        gt = utils.gt.gt.GT(filename='data/GT/Cam1_clip.avi.txt')
        utils.gt.visualize.visualize('/datagrid/ferda/data/ants_ist/camera_1/Cam1_clip.avi',
                                     'test/out/gt_visualize2.mp4',
                                     [gt, pose_gt],
                                     duration=1)
