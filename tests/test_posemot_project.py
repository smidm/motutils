import unittest
from utils.gt.posemot_project import PoseMotProject
from utils.gt.mot_project import MotProject
from core.project.project import Project


class PoseMotProjectTestCase(unittest.TestCase):
    def setUp(self):
        self.p = Project('test/project/Sowbug3_cut_300_frames')
        self.gt_filename = 'data/GT/Sowbug3_cut.txt'
        # self.gt = MotProject(filename='data/GT/Sowbug3_cut.txt')

    # def test_from_gt_and_regions(self):
    #     posegt = PoseMotProject.from_gt_and_regions(self.p, self.gt_filename)
    #     # import matplotlib.pylab as plt
    #     # frame = 9
    #     # plt.imshow(self.p.img_manager.get_whole_img(frame))
    #     # posegt.draw([frame])


if __name__ == '__main__':
    unittest.main()
