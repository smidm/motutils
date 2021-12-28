import unittest

import mot_utils
from mot_utils.io import load_any_mot, load_posemot_sleap_analysis


class IOTestCase(unittest.TestCase):
    def test_load_any_mot(self):
        gt = load_any_mot('tests/data/Sowbug3_cut_pose.csv')
        self.assertTrue(isinstance(gt, mot_utils.PoseMot))
        gt = load_any_mot('tests/data/Sowbug3_cut.txt')
        self.assertTrue(isinstance(gt, mot_utils.Mot))

    def test_load_posemot_sleap_analysis(self):
        tracks = load_posemot_sleap_analysis('tests/data/sample_sleap.analysis.h5')


if __name__ == '__main__':
    unittest.main()
