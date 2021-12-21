import unittest

import mot_utils
from mot_utils.io import load_any_mot, load_posemot_sleap_analysis


class IOTestCase(unittest.TestCase):
    def test_load_any_mot(self):
        gt = load_any_mot('data/GT/Sowbug3_cut_pose.csv')
        self.assertTrue(isinstance(gt, mot_utils.PoseMot))
        gt = load_any_mot('data/GT/5Zebrafish_nocover_22min.txt')
        self.assertTrue(isinstance(gt, mot_utils.Mot))

    def test_load_posemot_sleap_analysis(self):
        tracks = load_posemot_sleap_analysis('/home/matej/prace/ferda/behavior/src/data/HH2_crf17_pre.mp4.analysis.h5')


if __name__ == '__main__':
    unittest.main()
