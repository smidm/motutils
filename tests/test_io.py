import unittest

import motutils
from motutils.io import load_any_mot, load_posemot_sleap_analysis


class IOTestCase(unittest.TestCase):
    def test_load_any_mot(self):
        gt = load_any_mot('tests/data/Sowbug3_cut_pose.csv')
        self.assertTrue(isinstance(gt, motutils.PoseMot))
        gt = load_any_mot('tests/data/Sowbug3_cut.txt')
        self.assertTrue(isinstance(gt, motutils.Mot))

    def test_load_posemot_sleap_analysis(self):
        load_posemot_sleap_analysis('tests/data/sample_sleap.analysis.h5')


if __name__ == '__main__':
    unittest.main()
