import unittest
from numpy.testing import assert_array_almost_equal
import numpy as np
import utils.gt.mot
from shapes.bbox import BBox
import io
import xarray


class MotTestCase(unittest.TestCase):
    def setUp(self):
        self.filename = 'data/GT/Sowbug3_cut.txt'
        """
        1,1,434.48703703703706,279.04814814814813,-1,-1,1
        1,2,277.67721518987344,293.62025316455697,-1,-1,1
        1,3,179.2206866895768,407.8803152521979,-1,-1,1
        1,4,180.0,430.0,-1,-1,1
        1,5,154.97222222222223,397.0,-1,-1,1
        2,1,434.4667896678967,280.12546125461256,-1,-1,1
        2,2,278.1547231270358,293.6628664495114,-1,-1,1
        2,3,179.2206866895768,407.8803152521979,-1,-1,1
        2,4,180.0,430.0,-1,-1,1
        2,5,155.18852459016392,396.3098360655738,-1,-1,1
        3,1,434.61317254174395,280.8617810760668,-1,-1,1
        3,2,278.5096463022508,293.89549839228295,-1,-1,1
        3,3,179.2206866895768,407.8803152521979,-1,-1,1
        3,4,180.0,430.0,-1,-1,1
        3,5,155.21959459459458,396.02027027027026,-1,-1,1
        4,1,434.8387978142076,281.5255009107468,-1,-1,1
        4,2,278.8049180327869,293.89016393442625,-1,-1,1
        4,3,179.2206866895768,407.8803152521979,-1,-1,1
        4,4,180.0,430.0,-1,-1,1
        4,5,154.8202614379085,395.98366013071893,-1,-1,1
        5,1,434.7900900900901,282.10360360360363,-1,-1,1
        5,2,279.1031746031746,294.15396825396823,-1,-1,1
        5,3,179.2206866895768,407.8803152521979,-1,-1,1
        5,4,180.0,430.0,-1,-1,1
        ...
        """
        self.gt = utils.gt.mot.Mot(self.filename)

    def test_init_blank(self):
        self.gt.init_blank(list(range(0, 101)), list(range(1, 6)))
        print(self.gt.ds)

    def test_load(self):
        ds = self.gt.ds
        self.assertEqual(len(ds['frame']), 5928)
        self.assertTrue(np.all(ds['frame'][[0, -1]] == (0, 5927)))
        self.assertEqual(len(ds['id']), 5)

    def test_save(self):
        out_file = 'test/out/gttestcase.txt'
        self.gt.save(out_file)
        # self.assertTrue(filecmp.cmp(self.filename, out_file), 'saved file differs from source file')
        # # differs is float rounding and int / float

    def test_print_statistics(self):
        self.gt.print_statistics()

    def test_num_ids(self):
        self.assertEqual(self.gt.num_ids(), 5)

    def test_get_roi(self):
        print(self.gt.get_roi())

    def test_add_delta(self):
        # fr id x                   y
        # 1, 1, 434.48703703703706, 279.04814814814813, -1, -1, 1
        self.gt.add_delta(delta_x=-434)
        pos = self.gt.get_positions(frame=0).sel({'id': 1})
        self.assertAlmostEqual(pos['x'].item(), 0, 0)
        self.assertAlmostEqual(pos['y'].item(), 279, 0)

        self.gt.add_delta(delta_x=434)
        pos = self.gt.get_positions(frame=0).sel({'id': 1})
        self.assertAlmostEqual(pos['x'].item(), 434, 0)
        self.assertAlmostEqual(pos['y'].item(), 279, 0)

        self.gt.add_delta(delta_y=-279)
        pos = self.gt.get_positions(frame=0).sel({'id': 1})
        self.assertAlmostEqual(pos['x'].item(), 434, 0)
        self.assertAlmostEqual(pos['y'].item(), 0, 0)

        self.gt.add_delta(delta_y=279)
        pos = self.gt.get_positions(frame=0).sel({'id': 1})
        self.assertAlmostEqual(pos['x'].item(), 434, 0)
        self.assertAlmostEqual(pos['y'].item(), 279, 0)

        self.gt.add_delta(delta_frames=2)
        self.assertAlmostEqual(self.gt.min_frame(), 2)
        pos = self.gt.get_positions(frame=2).sel({'id': 1})
        self.assertAlmostEqual(pos['x'].item(), 434, 0)
        self.assertAlmostEqual(pos['y'].item(), 279, 0)

    def test_get_positions(self):
        frame_pos = self.gt.get_positions(1)
        """
        <xarray.Dataset>
        Dimensions:     (id: 5)
        Coordinates:
            frame       int64 1
          * id          (id) int64 1 2 3 4 5
        Data variables:
            x           (id) float64 434.5 278.2 179.2 180.0 155.2
            y           (id) float64 280.1 293.7 407.9 430.0 396.3
            confidence  (id) float64 1.0 1.0 1.0 1.0 1.0        
        """
        assert_array_almost_equal(frame_pos['x'],
                                  [434.46, 278.15, 179.22, 180.0, 155.18], 1)
        assert_array_almost_equal(frame_pos['y'],
                                  [280.12, 293.66, 407.88, 430.0, 396.30], 1)

        assert_array_almost_equal(frame_pos.sel({'id': 1})[['x', 'y']].to_array(),
                                  [434.46, 280.12], 1)

        self.assertAlmostEqual(frame_pos.sel({'id': 1})['x'].item(), 434.46, 1)
        self.assertAlmostEqual(frame_pos.sel({'id': 1})['y'].item(), 280.12, 1)

    def test_get_xy_numpy(self):
        xy = self.gt.get_xy_numpy(0)
        self.assertTupleEqual(xy.shape, (5, 2))

    def test_get_positions_dataframe(self):
        df = self.gt.get_positions_dataframe(1)
        """
            frame           x           y  width  height  confidence
        id                                                          
        1       1  434.466790  280.125461    NaN     NaN         1.0
        2       1  278.154723  293.662866    NaN     NaN         1.0
        3       1  179.220687  407.880315    NaN     NaN         1.0
        4       1  180.000000  430.000000    NaN     NaN         1.0
        5       1  155.188525  396.309836    NaN     NaN         1.0        
        """
        self.assertAlmostEqual(df.loc[1].x, 434.46, 1)
        self.assertAlmostEqual(df.loc[1].y, 280.12, 1)
        for pos_id, row in df.iterrows():
            self.assertEqual(pos_id, 1)
            self.assertAlmostEqual(row.x, 434.46, 1)
            self.assertAlmostEqual(row.y, 280.12, 1)
            break

    def test_get_bboxes(self):
        self.gt.bbox_size_px = 10
        bboxes = self.gt.get_bboxes(0)
        self.assertEqual(len(bboxes), 5)
        bbox = bboxes[-1]  # 1,5,154.97222222222223,397.0,-1,-1,1
        self.assertEqual(bbox.width, self.gt.bbox_size_px)
        self.assertEqual(bbox.height, self.gt.bbox_size_px)
        assert_array_almost_equal(bbox.xy, [154.97, 397], 1)
        self.assertAlmostEqual(bbox.xmin, 149.97, 1)
        self.assertAlmostEqual(bbox.xmax, 159.97, 1)
        self.assertAlmostEqual(bbox.ymin, 392, 1)
        self.assertAlmostEqual(bbox.ymax, 402, 1)

    def test_match_bbox(self):
        self.gt.bbox_size_px = 10
        gt_bbox = BBox.from_xycenter_wh(154.97, 397, 10, 10, frame=0)
        matched_bbox = self.gt.match_bbox(gt_bbox)
        self.assertEqual(matched_bbox.frame, 0)
        self.assertEqual(matched_bbox.obj_id, 5)

        no_match_bbox = BBox(1000, 1000, 1010, 1010, 0)
        matched_bbox = self.gt.match_bbox(no_match_bbox)
        self.assertTrue(matched_bbox is None)

    def test_get_matching_obj_id(self):
        self.gt.bbox_size_px = 10
        gt_bbox = BBox.from_xycenter_wh(154.97, 397, 10, 10, frame=0)
        obj_id = self.gt.get_matching_obj_id(gt_bbox)
        self.assertEqual(obj_id, 5)

        no_match_bbox = BBox(1000, 1000, 1010, 1010, 0)
        obj_id = self.gt.get_matching_obj_id(no_match_bbox)
        self.assertTrue(obj_id is None)

    def test_set_position(self):
        self.gt.set_position(0, 1, 1000, 2000)
        df = self.gt.get_positions_dataframe(0)
        self.assertEqual(df.loc[1].x, 1000)
        self.assertEqual(df.loc[1].y, 2000)

    def test_match_xy(self):
        ds = self.gt.match_xy(0, (1000, 1000), 100)
        self.assertTrue(ds is None)
        ds = self.gt.match_xy(0, (180, 430), 1)
        self.assertEqual(ds.id, 4)
        """
        <xarray.Dataset>
        Dimensions:     (id: 5)
        Coordinates:
            frame       int64 100
          * id          (id) int64 1 2 3 4 5
        Data variables:
            x           (id) float64 430.4 294.0 182.2 163.4 123.9
            y           (id) float64 309.8 299.2 403.7 421.7 382.6
            confidence  (id) float64 1.0 1.0 1.0 1.0 1.0
        """
        ds = self.gt.match_xy(100, (180, 430), 1)
        self.assertTrue(ds is None)

    def test_minmax_frame(self):
        ret = self.gt.min_frame()
        print(ret)
        ret = self.gt.max_frame()
        print(ret)

    def test_interpolate_positions(self):
        ds = self.gt.interpolate_positions()
        self.assertEqual(len(ds['frame']), 5928)
        self.assertEqual(len(ds['id']), 5)

        ds = self.gt.interpolate_positions(frames=[0, 1, 2, 3])
        self.assertEqual(len(ds['frame']), 4)
        self.assertEqual(len(ds['id']), 5)

        ds = self.gt.interpolate_positions(ids=[1, 4])
        self.assertEqual(len(ds['frame']), 5928)
        self.assertEqual(len(ds['id']), 2)

        ds = self.gt.interpolate_positions(frames=[0, 1], ids=[1, 2, 3])
        self.assertEqual(len(ds['frame']), 2)
        self.assertEqual(len(ds['id']), 3)

        self.gt.set_position(10, 1, 10, 10)
        self.gt.set_position(11, 1, np.nan, np.nan)
        self.gt.set_position(12, 1, 20, 30)
        ds = self.gt.interpolate_positions(frames=[11], ids=[1])
        self.assertEqual(ds.x, 15)
        self.assertEqual(ds.y, 20)

    def test_get_missing_positions(self):
        ret = self.gt.get_missing_positions()
        print(ret)

    def test_draw(self):
        self.gt.draw([0])

    def test_find_mapping(self):
        csv_str = """frame,id,x,y,width,height,confidence
                1,1,277.67721518987344,293.62025316455697,-1,-1,1
                1,2,434.48703703703706,279.04814814814813,-1,-1,1
                1,3,179.2206866895768,407.8803152521979,-1,-1,1
                1,4,180.0,430.0,-1,-1,1
                1,5,154.97222222222223,397.0,-1,-1,1
                2,1,278.1547231270358,293.6628664495114,-1,-1,1
                2,2,434.4667896678967,280.12546125461256,-1,-1,1
                2,3,179.2206866895768,407.8803152521979,-1,-1,1
                2,4,180.0,430.0,-1,-1,1
                2,5,155.18852459016392,396.3098360655738,-1,-1,1
        """
        csv_file = io.StringIO(csv_str)
        other = utils.gt.mot.Mot(csv_file)
        mapping = self.gt.find_mapping(other)
        self.assertEqual(mapping, {1: 2, 2: 1, 3: 3, 4: 4, 5: 5})

    def test_get_object_distance(self):
        dist = self.gt.get_object_distance(0, 1, self.gt.get_object(10, 1))


if __name__ == '__main__':
    unittest.main()
