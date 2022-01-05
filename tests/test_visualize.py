import unittest

import motutils


class VisualizeTestCase(unittest.TestCase):
    def test_visualize(self):
        gt = motutils.Mot("tests/data/Sowbug3_cut.csv")
        motutils.visualize(
            "tests/data/Sowbug3_cut.mp4",
            "tests/out/Sowbug3_cut_visualize.mp4",
            [gt.draw_frame],
            duration=1,
        )

    def test_visualize_posemot(self):
        pose_gt = motutils.PoseMot(filename_or_buffer="tests/data/Sowbug3_cut_pose.csv")
        gt = motutils.Mot(filename_or_buffer="tests/data/Sowbug3_cut.csv")
        motutils.visualize(
            "tests/data/Sowbug3_cut.mp4",
            "tests/out/Sowbug3_cut_visualize_pose.mp4",
            [gt.draw_frame, pose_gt.draw_frame],
            duration=0.5,
        )
