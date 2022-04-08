import unittest

import motutils
from motutils import io


class IOTestCase(unittest.TestCase):
    def test_load_any_mot(self):
        gt = io.load_any_mot("tests/data/Sowbug3_cut_pose.csv")
        self.assertTrue(isinstance(gt, motutils.PoseMot))
        self.assertEqual(gt.num_ids(), 5)
        gt = io.load_any_mot("tests/data/Sowbug3_cut.csv")
        self.assertTrue(isinstance(gt, motutils.Mot))
        self.assertEqual(gt.num_ids(), 5)

    def test_load_posemot_sleap_analysis(self):
        io.load_sleap_analysis_as_posemot("tests/data/sample_sleap.analysis.h5")

    def test_load_idtracker(self):
        df = io.load_idtracker("tests/data/idtracker/trajectories.txt")
        self.assertEqual(
            df.frame.max(), 4499
        )  # trajectories.txt somehow omitted last frame
        self.assertEqual(len(df.id.unique()), 5)
        df = df.dropna(subset=["x", "y"])
        self.assertTrue((df.x < 640).all() and (df.y < 640).all())

    def test_load_idtrackerai(self):
        df = io.load_idtrackerai("tests/data/idtrackerai/trajectories_wo_gaps.npy")
        self.assertEqual(df.frame.max(), 4500)
        self.assertEqual(len(df.id.unique()), 5)
        self.assertTrue((df.x < 640).all() and (df.y < 640).all())

    def test_load_toxtrac(self):
        df = io.load_toxtrac("tests/data/toxtrac/Tracking_0.txt")
        self.assertEqual(len(df.id.unique()), 5)
        self.assertTrue((df.x < 640).all() and (df.y < 640).all())


if __name__ == "__main__":
    unittest.main()
