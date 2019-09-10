from utils.gt.gt_project import GtProject, GtReid
from core.project.project import Project
import unittest
from numpy.testing import assert_array_almost_equal
import numpy as np
from core.region.region import Region


class GtProjectTestCase(unittest.TestCase):
    def setUp(self):
        self.p = Project('test/project/Sowbug3_cut_300_frames')
        self.gt = GtProject(filename='data/GT/Sowbug3_cut.txt')
        """
        <xarray.Dataset>
        Dimensions:     (id: 5)
        Coordinates:
            frame       int64 0
          * id          (id) int64 1 2 3 4 5
        Data variables:
            x           (id) float64 434.5 277.7 179.2 180.0 155.0
            y           (id) float64 279.0 293.6 407.9 430.0 397.0        
        
        <xarray.Dataset>
        Dimensions:     (id: 5)
        Coordinates:
            frame       int64 1
          * id          (id) int64 1 2 3 4 5
        Data variables:
            x           (id) float64 434.5 278.2 179.2 180.0 155.2
            y           (id) float64 280.1 293.7 407.9 430.0 396.3
        
        <xarray.Dataset>
        Dimensions:     (id: 5)
        Coordinates:
            frame       int64 2
          * id          (id) int64 1 2 3 4 5
        Data variables:
            x           (id) float64 434.6 278.5 179.2 180.0 155.2
            y           (id) float64 280.9 293.9 407.9 430.0 396.0
        """

    def test_from_tracklets(self):
        gt_tracklets = GtProject.from_tracklets(self.p)
        """
        gt_tracklets.ds.loc[dict(frame=0)].dropna(dim='id', how='all')

        <xarray.Dataset>
        Dimensions:     (id: 4)
        Coordinates:
            frame       int64 0
          * id          (id) int64 1 2 4 5
        Data variables:
            y           (id) float64 226.2 241.6 364.5 346.0
            x           (id) float64 388.6 231.4 134.8 108.9
            
        <xarray.Dataset>
        Dimensions:     (id: 4)
        Coordinates:
            frame       int64 1
          * id          (id) int64 1 2 4 5
        Data variables:
            y           (id) float64 227.3 241.6 364.5 345.5
            x           (id) float64 388.5 231.5 134.7 108.9

        <xarray.Dataset>
        Dimensions:     (id: 4)
        Coordinates:
            frame       int64 2
          * id          (id) int64 1 2 4 5
        Data variables:
            y           (id) float64 228.3 241.6 364.6 345.3
            x           (id) float64 388.4 232.0 134.6 108.7        
        """

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

    def test_match_on_data(self):
        self.gt.set_project_offsets(self.p)
        match = self.gt.match_on_data(self.p, frames=(0, 3))
        self.assertEqual(len(match), 2)
        self.assertEqual(match.keys(), [0, 3])

        match = self.gt.match_on_data(self.p, frames=(0, 3), match_on='regions')
        self.assertEqual(len(match), 2)
        self.assertEqual(match.keys(), [0, 3])

        # # visualize gt and tracklets
        # pos = GtProject.from_tracklets(self.p)
        # pos.draw([0], marker='^')
        # self.gt.draw([0], marker='v')
        # import matplotlib.pylab as plt
        # plt.legend()
        # plt.show()

    def test_tracklet_id_set(self):
        self.gt.set_project_offsets(self.p)
        self.gt.tracklet_id_set(next(self.p.chm.chunk_gen()), self.p)

    def test_get_tracklet_cardinality(self):
        self.gt.set_project_offsets(self.p)
        tracklet = next(self.p.chm.chunk_gen())
        cardinality = self.gt.get_tracklet_cardinality(self.p, tracklet)
        self.assertEqual(cardinality, 1)

    def test_fill_tracklet_cardinalites(self):
        self.gt.set_project_offsets(self.p)
        # tracklet_cardinalities_pre = {t.id(): t.cardinality for t in self.p.chm.tracklet_gen()}
        tracklet_segmentation_class_pre = {t.id(): t.segmentation_class for t in self.p.chm.tracklet_gen()}
        self.gt.fill_tracklet_cardinalites(self.p)
        tracklet_cardinalities_post = {t.id(): t.cardinality for t in self.p.chm.tracklet_gen()}
        tracklet_segmentation_class_post = {t.id(): t.segmentation_class for t in self.p.chm.tracklet_gen()}

    def test_get_single_region_ids(self):
        self.gt.set_project_offsets(self.p)
        single_region_ids, animal_ids = self.gt.get_single_region_ids(self.p, max_frame=10)
        pass

    def test_get_cardinalities(self):
        self.gt.set_project_offsets(self.p)
        cardinalities = self.gt.get_regions_cardinalities(self.p, 0)
        pass

    def test_get_region_cardinality(self):
        self.gt.set_project_offsets(self.p)
        card = self.gt.get_region_cardinality(self.p, self.p.rm[1])
        pass

    def test_get_cardinalities_without_project(self):
        region = Region(frame=0)
        region.set_centroid([279, 434])
        cardinalities = self.gt.get_regions_cardinalities_without_project([region], 2)
        cardinalities = self.gt.get_regions_cardinalities_without_project([region], 0.5)
        pass
