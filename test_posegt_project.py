import unittest
from utils.gt.posegt_project import PoseGtProject
from utils.gt.gt_project import GtProject
from core.project.project import Project


class PoseGtProjectTestCase(unittest.TestCase):
    def setUp(self):
        self.p = Project('test/project/Sowbug3_cut_300_frames')
        self.gt_filename = 'data/GT/Sowbug3_cut.txt'
        # self.gt = GtProject(filename='data/GT/Sowbug3_cut.txt')

    # def test_from_gt_and_regions(self):
    #     posegt = PoseGtProject.from_gt_and_regions(self.p, self.gt_filename)
    #     # import matplotlib.pylab as plt
    #     # frame = 9
    #     # plt.imshow(self.p.img_manager.get_whole_img(frame))
    #     # posegt.draw([frame])


if __name__ == '__main__':
    unittest.main()
