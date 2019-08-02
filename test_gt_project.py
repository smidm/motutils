from utils.gt.gt import GT
from utils.gt.gt_project import GtProjectMixin, GtDummyReIdMixin
from core.project.project import Project
import unittest
from numpy.testing import assert_array_almost_equal


class GtProject(GtProjectMixin, GT):
    pass


class GtReid(GtDummyReIdMixin, GT):
    pass


class GtProjectTestCase(unittest.TestCase):
    def setUp(self):
        self.p = Project('test/project/Sowbug3_cut_300_frames')
        self.gt = GtProject('data/GT/Sowbug3_cut.txt')

    def test_set_project_offsets(self):
        min_frame = self.gt.min_frame()
        pos = self.gt.get_positions(501).copy(deep=True)

        self.p.video_crop_model['x1'] = 101
        self.p.video_crop_model['y1'] = 102
        self.p.video_start_t = 501

        self.gt.set_project_offsets(self.p)

        pos0 = self.gt.get_positions(0)
        assert_array_almost_equal(pos.x - 101, pos0.x)
        assert_array_almost_equal(pos.y - 102, pos0.y)
        self.assertEqual(min_frame - 501, self.gt.min_frame())

    # def test_match_on_data(self):
    #     self.gt.match_on_data(self.p)


