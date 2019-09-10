import unittest
import numpy as np
from utils.gt.posegt import PoseGt, load_any
from utils.gt.gt import GT


class PoseGtTestCase(unittest.TestCase):
    def setUp(self):
        self.gt = PoseGt()
        self.gt.init_blank(range(3), range(2), n_points=3)
        for frame in self.gt.ds.frame.values:
            for obj_id in self.gt.ds.id.values:
                for keypoint in self.gt.ds.keypoint.values:
                    val = 10 * frame + obj_id + 0.1 * keypoint
                    self.gt.set_position(frame, obj_id, keypoint, val, val)

    def test_init_blank(self):
        self.gt.init_blank(frames=range(4), ids=range(3), n_points=2)
        self.assertTrue('x' in self.gt.ds)
        self.assertTrue('y' in self.gt.ds)
        self.assertEqual(self.gt.ds['x'].shape, (4, 3, 2))
        self.assertEqual(self.gt.ds.sel(dict(frame=0, id=0))['x'].shape, (2,))

    def test_load_save(self):
        out_file = 'test/out/gttestcase.txt'
        self.gt.save(out_file)
        loaded_gt = PoseGt()
        loaded_gt.load(out_file)
        self.assertEqual(self.gt.ds, loaded_gt.ds)

    def test_get_obj_roi(self):
        xmin, xmax, ymin, ymax = self.gt.get_obj_roi(1, 1)
        self.assertEqual(xmin, 11.0)
        self.assertEqual(xmax, 11.2)
        self.assertEqual(ymin, 11.0)
        self.assertEqual(ymax, 11.2)

        # single coordinate is nan, nans are ignored
        self.gt.set_position(frame=1, obj_id=1, keypoint=0, x=np.nan, y=1)
        xmin, xmax, ymin, ymax = self.gt.get_obj_roi(frame=1, obj_id=1)
        self.assertEqual(xmin, 11.1)
        self.assertEqual(xmax, 11.2)
        self.assertEqual(ymin, 1)
        self.assertEqual(ymax, 11.2)

        # all x coordinates are nan, xmin xmax are nan
        self.gt.set_position(frame=1, obj_id=1, keypoint=1, x=np.nan, y=1)
        self.gt.set_position(frame=1, obj_id=1, keypoint=2, x=np.nan, y=1)
        xmin, xmax, ymin, ymax = self.gt.get_obj_roi(frame=1, obj_id=1)
        self.assertTrue(np.isnan(xmin))
        self.assertTrue(np.isnan(xmax))
        self.assertEqual(ymin, 1)
        self.assertEqual(ymax, 1)

    def test_get_bboxes(self):
        bboxes = self.gt.get_bboxes(0)
        self.assertEqual(len(bboxes), 2)
        self.assertEqual(bboxes[0].frame, 0)
        self.assertEqual(bboxes[1].frame, 0)

        # all x keypoints are nan, obj_id == 1 won't give a bbox
        self.gt.set_position(frame=0, obj_id=1, keypoint=0, x=np.nan, y=1)
        self.gt.set_position(frame=0, obj_id=1, keypoint=1, x=np.nan, y=1)
        self.gt.set_position(frame=0, obj_id=1, keypoint=2, x=np.nan, y=1)
        bboxes = self.gt.get_bboxes(0)
        self.assertEqual(len(bboxes), 1)

    def test_match_xy(self):
        xy = np.array([[0, 0],
                       [0.1, 0.1],
                       [0.2, 0.2]])  # shape (n_keypoints, 2)
        match = self.gt.match_xy(0, xy)
        self.assertEqual(match.frame, 0)
        self.assertEqual(match.id, 0)

        xy = np.array([[1, 1],
                       [1.1, 1.1],
                       [1.2, 1.2]])  # shape (n_keypoints, 2)
        match = self.gt.match_xy(0, xy)
        self.assertEqual(match.frame, 0)
        self.assertEqual(match.id, 1)

        xy = np.array([[2, 2],
                       [2.1, 2.1],
                       [2.2, 2.2]])  # shape (n_keypoints, 2)
        match = self.gt.match_xy(0, xy)
        self.assertEqual(match.frame, 0)
        self.assertEqual(match.id, 1)

        match = self.gt.match_xy(0, xy, max_match_distance_px=np.sqrt(2) + 0.1)
        self.assertEqual(match.frame, 0)
        self.assertEqual(match.id, 1)

        match = self.gt.match_xy(0, xy, max_match_distance_px=np.sqrt(2) - 0.1)
        self.assertEqual(match, None)

        xy = np.array([[2, 1],
                       [1.1, 1.1],
                       [1.2, 1.2]])  # shape (n_keypoints, 2)
        match = self.gt.match_xy(0, xy, max_match_distance_px=1)
        self.assertEqual(match.frame, 0)
        self.assertEqual(match.id, 1)

        match = self.gt.match_xy(0, xy, max_match_distance_px=0.9)
        self.assertEqual(match, None)

    # def test_draw(self):
    #     import matplotlib.pylab as plt
    #     self.gt.draw(frames=[0])
    #     plt.legend()
    #     plt.show()
    #
    #     self.gt.draw(frames=[0], ids=[0])
    #     plt.legend()
    #     plt.show()
    #
    #     self.gt.draw()
    #     plt.legend()
    #     plt.show()

    def test_load_any(self):
        out_file = 'test/out/posegt_test_load_any.txt'
        self.gt.save(out_file)
        gt = load_any(out_file)
        self.assertTrue(isinstance(gt, PoseGt))
        gt = load_any('data/GT/5Zebrafish_nocover_22min.txt')
        self.assertTrue(isinstance(gt, GT))


if __name__ == '__main__':
    unittest.main()
