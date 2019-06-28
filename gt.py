from __future__ import print_function
import pickle
import sys
import warnings
import numpy as np
import tqdm
from scipy.spatial.distance import cdist
import pandas as pd


class GT(object):
    """
    Ground truth handling.

    When working with a FERDA project, don't forget to set spatial and temporal offsets, see set_project_offsets().

    None means not defined.

    """
    def __init__(self, filename=None):
        """
        ground truth stored in
        pandas.DataFrame indexed by frame and id, frames 0-indexed:

                           x           y  width  height  confidence
        frame id
        0     1   211.479849  477.982368     -1      -1           1
              2   142.146532  491.562640     -1      -1           1
              3   257.125920  512.797496     -1      -1           1
              4   231.636725  656.953100     -1      -1           1
              5   261.582812  592.857813     -1      -1           1
        """
        self.num_ids = 0
        self.df = None

        self.bbox_size_px = None
        self.bbox_match_minimal_iou = 0.5

        if filename is not None:
            self.load(filename)

        super(GT, self).__init__()  # this calls potential mixin classes init methods
                                    # see https://stackoverflow.com/a/6099026/322468

    def init_blank(self, num_ids, frame_range):
        """
        Initialize blank ground truth.

        :param num_ids: number of identities
        :param frame_range: start frame, end frame
        """
        self.num_ids = num_ids
        index = pd.MultiIndex.from_product([xrange(frame_range[0], frame_range[1]), xrange(1, self.num_ids + 1)],
                                           names=['frame', 'id'])
        self.df = pd.DataFrame(columns=['x', 'y', 'width', 'height', 'confidence'], index=index)

    def load(self, filename):
        """
        Load Multiple Object Tacking Challenge trajectories file.

        Format described in https://arxiv.org/abs/1603.00831, section 3.3 Data Format

        Loads trajectories into a DataFrame, columns frame and id start with 1 (MATLAB indexing).

        :param filename: mot filename
        """
        df = pd.read_csv(filename, names=[u'frame', u'id', u'x', u'y', u'width', u'height', u'confidence'],
                         index_col=[u'frame', u'id'], converters={u'frame': lambda x: int(x) - 1})
        self.df = df[(df.x != -1) & (df.y != -1)]
        self.num_ids = df.index.unique('id')

    def print_statistics(self):
        number_of_ids_counts = self.df.count(axis=0, level='frame')['x'].value_counts()
        print('counts of number of object ids in frames:')
        print(number_of_ids_counts)

        print('frames with number of objects other than 0 or num_ids:')
        anomalous_object_counts = set(number_of_ids_counts.index) - {0, self.num_ids}
        for count in anomalous_object_counts:
            print(self.df.loc[self.df.index.levels[0][self.df.count(axis=0, level='frame')['x'] == count]])    def get_roi(self):
        """
        Return GT rectangular bounds.

        :return: xmin, xmax, ymin, ymax
        """
        return self.df['x'].min(), self.df['x'].max(), self.df['y'].min(), self.df['y'].max()

    def set_project_offsets(self, project):
        self.shift_gt(
            -project.video_crop_model['x1'] if project.video_crop_model is not None else 0,
            -project.video_crop_model['y1'] if project.video_crop_model is not None else 0,
            -project.video_start_t)

    def shift_gt(self, delta_x=0, delta_y=0, delta_frames=0):
        """
        Shift ground truth positions and frame numbers by deltas.
        """
        self.df['x'] += delta_x
        self.df['y'] += delta_y
        frames_index = gt.df.index.levels[0] + delta_frames
        self.df.index = pd.MultiIndex.from_product([frames_index, self.df.index.levels[1]], names=['frame', 'id'])

    def get_clear_positions(self, frame):
        frame -= self.__frames_offset
        p = [None for _ in range(self.num_ids)]
        if frame in self.__positions:
            for i, it in enumerate(self.__positions[frame]):
                if it is not None:
                    # y, x, type_ = it
                    y, x = it
                    # if type_ == 1:
                    p[i] = (y + self.__y_offset, x + self.__x_offset)
                    # p[self.__permutation[i]] = (y + self.__y_offset, x + self.__x_offset)

        return p

    def get_positions(self, frame):
        """

        :param frame:
        :return: DataFrame, indexed by id, with columns x, y, width, height, confidence
        """
        return self.df.loc[frame]

    def get_positions_numpy(self, frame):
        """

        :param frame:
        :return: ndarray, shape=(n, 2)
        """
        return self.get_positions()[['x', 'y']].to_numpy()

    def get_bboxes(self, frame):
        """
        Get GT bounding boxes in a frame.

        :param frame: frame number
        :return: list of bounding boxes (BBox)
        """
        assert 'bbox_size_px' in dir(self) and self.bbox_size_px is not None
        bboxes = []
        for obj_id, obj in self.df.loc[frame].iterrows():
            if obj.width == -1 or obj.height == -1:
                width = self.bbox_size_px
                height = self.bbox_size_px
            else:
                width = obj.width
                height = obj.height
            bbox = BBox.from_xycenter_hw(obj.x, obj.y, width, height, frame)
            bbox.obj_id = obj_id
            bboxes.append(bbox)
        return bboxes

    def match_bbox(self, query_bbox):
        """
        Match query bounding box to the ground truth.

        :param query_bbox: BBox with defined frame
        :return: None if false positive, best matching BBox otherwise
        """
        bboxes = self.get_bboxes(query_bbox.frame)
        ious = np.array([bbox.iou(query_bbox) for bbox in bboxes])
        if ious.max() < self.bbox_match_minimal_iou:
            return None  # fp
        else:
            return bboxes[ious.argmax()]

    def get_matching_obj_id(self, query_bbox):
        """
        Match query bounding box to the ground truth and return the matching gt object id.

        :param query_bbox:
        :return: object id or None
        """
        matching_bbox = self.match_bbox(query_bbox)
        if matching_bbox is not None:
            return matching_bbox.obj_id
        else:
            return None

    def set_position(self, frame, id_, x, y):
        self.df.loc[(frame, id_), ['x', 'y']] = (x, y)

    def save(self, filename, make_backup=False):
        import os
        import datetime

        if make_backup and os.path.exists(filename):
            dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            os.rename(filename, filename[:-4] + '_' + dt + '.txt')

        self.df.to_csv(filename, header=False, index=False)

    def match_xy(self, frame, xy, maximal_match_distance=None):
        """
        Match query xy to the ground truth.

        :param xy: tuple
        :return: None if false positive, best matching gt row
        """
        distance_vectors = self.df.loc[frame][['x', 'y']] - xy
        distances = np.sqrt((distance_vectors['x']**2 + distance_vectors['y']**2))
        matching_id = distances.idxmin()
        if distances[matching_id] > maximal_match_distance:
            return None  # fp
        else:
            return self.df.loc[frame, matching_id]

    def min_frame(self):
        return self.df.index.levels[0].min()

    def max_frame(self):
        return self.df.index.levels[0].max()

    def match_on_data(self, project, frames=None, max_d=5, data_centroids=None, match_on='tracklets', permute=False,
                      progress=True):
        """
        Match ground truth on tracklets or regions.

        :param project: Project() instance
        :param frames: list or None for all frames where gt is defined
        :param max_d: maximum euclidean distance in px to match
        :param data_centroids: centroids for tracklets or regions, None to compute
        :param match_on: 'tracklets' or 'regions'
        :param permute:
        :return: match, match[frame][gt position id]: chunk or region id
        """
        from itertools import izip

        # num_frames = self.max_frame() - self.min_frame()

        not_matched = []
        match = {}
        i = 0

        if frames is None:
            frames = range(self.min_frame(), self.max_frame() + 1)

        for frame in tqdm.tqdm(frames, disable=not progress):
            match[frame] = [None for _ in range(len(project.animals))]

            # add chunk ids
            if match_on == 'tracklets':
                regions = []
                tracklet_ids = []
                regions_tracklets = project.gm.regions_and_t_ids_in_t(frame)
                for rid, t_id in regions_tracklets:
                    r = project.rm[rid]
                    if not r.is_origin_interaction():
                        regions.append(r)
                        tracklet_ids.append(t_id)
            else:
                regions = project.gm.regions_in_t(frame)

            # if len(regions) == 0:
            #     continue

            if data_centroids is None:
                centroids = np.array([r.centroid() for r in regions])
            else:
                centroids = data_centroids[frame]

            if len(centroids) == 0:
                continue

            pos = self.__positions[frame - self.__frames_offset]
            if None in pos:
                continue
            pos = np.array([(x[0] + self.__y_offset, x[1] + self.__x_offset) for x in pos])
            # pos = np.array([(y + self.__y_offset, x + self.__x_offset) for y, x in pos])  # check and replace line above

            centroids[np.isnan(centroids)] = np.inf
            try:
                dists = cdist(pos, centroids)
            except:
                print(centroids, regions, frame)

            m1_i = np.argmin(dists, axis=1)
            m1 = dists[range(pos.shape[0]), m1_i]

            for a_id, id_ in enumerate(m1_i):
                if permute:
                    a_id = self.__permutation[a_id]

                if m1[a_id] > max_d:
                    # try if inside region...
                    if match_on == 'tracklets':
                        for r, t_id in izip(regions, tracklet_ids):
                            if r.is_inside(pos[a_id], tolerance=max_d):
                                match[frame][a_id] = t_id
                                break
                    elif match_on == 'centroids':
                        raise Exception('not implemented')
                    else:
                        for r in regions:
                            if r.is_inside(pos[a_id], tolerance=max_d):
                                match[frame][a_id] = r.id()
                                break

                    if match[frame][a_id] is None:
                        not_matched.append(frame)
                else:
                    if match_on == 'tracklets':
                        match[frame][a_id] = tracklet_ids[id_]
                    elif match_on == 'centroids':
                        match[frame][a_id] = id_
                    else:
                        match[frame][a_id] = regions[id_].id()

            # TODO: solve big distances for oversegmented regions
            # dists[range(pos.shape[0]), m1_i] = np.inf
            # m2 = np.min(dists, axis=1)

            i += 1

            # if i % 10 == 0:
            #     print_progress(i, num_frames)

        # print "Done.."
        # print "Not matched in frames ", not_matched

        return match

    def tracklet_id_set_without_checks(self, tracklet, project):
        match = self.match_on_data(project, range(tracklet.start_frame(),
                                                  tracklet.end_frame() + 1, 10))

        keys = sorted([k for k in match.iterkeys()])
        match = [match[k] for k in keys]

        ids = self.__get_ids_from_match(match[0], tracklet.id())
        return [self.__permutation[id_] for id_ in ids]

    def tracklet_id_set(self, tracklet, project):
        """


        Args:
            tracklet:
            project:

        Returns:

        """
        match = self.match_on_data(project, range(tracklet.start_frame(),
                                                  tracklet.end_frame() + 1),
                                   progress=False)

        keys = sorted([k for k in match.iterkeys()])
        match = [match[k] for k in keys]

        ids = self.__get_ids_from_match(match[0], tracklet.id())
        if self.test_tracklet_consistency(tracklet, match, ids):
            return [self.__gt_id_to_real_permutation[id_] for id_ in ids]
        else:
            warnings.warn('Tracklet id: {} is inconsistent.'.format(tracklet.id()))
            print(match, ids)
            return None

    def __get_ids_from_match(self, match, t_id):
        return set([id_ for id_, x in enumerate(match) if x == t_id])

    def test_tracklet_consistency(self, tracklet, match, ids=None):
        if ids is None:
            ids = self.__get_ids_from_match(match[0], tracklet.id())

        for i in range(1, len(match)):
            if ids != self.__get_ids_from_match(match[i], tracklet.id()):
                print("CONSISTENCY I, ", i)

                if self.break_on_inconsistency:
                    return False

        return True

    def __match_mapping_possible(self, match, ids):
        # there is the same tracklet id in match[ids]
        t_ids = set([match[id_] for id_ in ids])
        if len(t_ids) != 1:
            return False

        t_id = list(t_ids)[0]
        # trackle id is only on match[ids] positions
        if len(self.__get_ids_from_match(match, t_id)) != len(ids):
            return False

        return True

    def test_tracklet_max_len(self, tracklet, project):
        """
        if any expansions results in tracklet rules violation, tracklet is of max length
        Returns:

        """

        ids = self.tracklet_id_set(tracklet, project)
        if ids is None:
            return False

        frame = tracklet.start_frame() - 1
        if frame > 0:
            match = [x for x in self.match_on_data(project, [frame]).itervalues()][0]
            if self.__match_mapping_possible(match, ids):
                return False

        frame = tracklet.end_frame() + 1
        if frame < self.__max_frame:
            match = [x for x in self.match_on_data(project, [frame]).itervalues()][0]
            if self.__match_mapping_possible(match, ids):
                return False

        return True

    def project_stats(self, p):
        num_max_len = 0
        num_max_len_singles = 0
        num_singles = 0
        not_consistent = 0

        not_consistent_list = []
        singles_splits = set()

        matches = self.match_on_data(p, frames=range(0, p.gm.end_t + 1))

        for t in p.chm.chunk_gen():
            single = False

            match = [matches[frame] for frame in range(t.start_frame(), t.end_frame() + 1)]
            # match = [x for x in self.match_on_data(p, frames=range(t.start_frame(p.gm), t.end_frame(p.gm) + 1)).itervalues()]
            if match[0].count(t.id()) == 1:
                single = True
                num_singles += 1

                singles_splits.add(t.start_frame())
                singles_splits.add(t.end_frame())

            if not self.test_tracklet_consistency(t, match):
                not_consistent += 1
                not_consistent_list.append(t)

            if self.test_tracklet_max_len(t, p):
                num_max_len += 1

                if single:
                    num_max_len_singles += 1
                    singles_splits.discard(t.start_frame())
                    singles_splits.discard(t.end_frame())

        num_t = len(p.chm)
        print("#max_len singles: {}({:.2%}) all: {}({:.2%}), #not consitent: {}".format(num_max_len_singles,
                                                                                        num_max_len_singles / float(
                                                                                            num_singles),
                                                                                        num_max_len,
                                                                                        num_max_len / float(num_t),
                                                                                        not_consistent)  \
            + "single nonmax splits:"  \
            + sorted(list(singles_splits)))

    def get_single_region_ids(self, project, max_frame=np.inf):
        single_region_ids = []
        animal_ids = []
        match = self.match_on_data(project, match_on='regions')

        for frame in match.iterkeys():
            if frame > max_frame:
                continue

            for a_id, r_id in enumerate(match[frame]):
                if r_id is None:
                    continue

                if match[frame].count(r_id) == 1:
                    single_region_ids.append(r_id)
                    animal_ids.append(a_id)

        return single_region_ids, animal_ids

    def segmentation_class_from_idset(self, idset):
        # def is_single(self):
        #     return self.segmentation_class == 0
        #
        # def is_multi(self):
        #     return self.segmentation_class == 1
        #
        # def is_noise(self):
        #     return self.segmentation_class == 2
        #
        # def is_part(self):
        #     return self.segmentation_class == 3
        #
        # def is_undefined(self):
        #     return self.segmentation_class == -1

        if len(idset) == 1 and idset[0] is not None:
            return 0

        if len(idset) > 1:
            return 1

        if len(idset) == 0:
            return 2

        if len(idset) == 1 and idset[0] is None:
            return 3

        return -1

    def get_class_and_id(self, tracklet, project, verbose=0):
        if verbose:
            print("ASKING ABOUT ID ", tracklet.id())

        id_set = self.tracklet_id_set(tracklet, project)
        t_class = self.segmentation_class_from_idset(id_set)

        return t_class, id_set

    def get_cardinalities(self, project, frame):
        """
        Get cardinalities for regions in the frame according to GT.

        :param project: core.project.Project instance
        :param frame: frame number
        :return: cardinalities, dict {region id: cardinality class, ...}, {42: 'single', ... }
        """
        match = self.match_on_data(project, [frame], match_on='regions', progress=False)
        region_ids, counts = np.unique(match[frame], return_counts=True)
        cardinalities = {}
        for rid, count in zip(region_ids, counts):
            if count == 1:
                cardinalities[rid] = 'single'
            elif count > 1:
                cardinalities[rid] = 'multi'
            elif count == 0:
                cardinalities[rid] = 'noise'
            else:
                assert True
        return cardinalities

    def get_positions_backwards(self, frame, max_frames_backwards=10):
        """
        Get defined positions in a frame and search for the rest backwards.

        :param frame: frame number
        :param max_frames_backwards: maximum frame distance from "frame"
        :return: positions DataFrame
        """
        positions = self.get_positions(frame)
        missing_ids = list(self.df.index.levels[1].difference(positions.index))

        for prev_frame in reversed(range(max(0, frame - max_frames_backwards), frame)):
            if not missing_ids:
                break
            prev_positions = self.get_positions(prev_frame)
            for obj_id in missing_ids:
                if obj_id in prev_positions.index:
                    positions.loc[obj_id] = prev_positions.loc[obj_id]
                    missing_ids.remove(obj_id)

        return positions.sort_index()


    def get_region_cardinality(self, project, region):
        """
        Get cardinality for a region according to GT.

        The region has to be included in the project region manager.

        :param project: core.project.project.Project instance
        :param region: core.region.region.Region instance
        :return: str, cardinality class, one of 'single', 'multi', 'noise'
        """
        assert region in self.project.rm
        assert region == self.project.rm[region.id()]
        cardinalities = self.get_cardinalities(project, region.frame())
        return cardinalities[region.id()]

    def get_cardinalities_without_project(self, regions, thresh_px):
        """
        Get cardinalities for regions independent on a project.

        Requires complete set of regions for a frame!

        :param regions: list of Region objects
        :param thresh_px: maximum distance between region and ground truth for valid match
        :return:
        """
        assert len(regions) > 0
        frame = regions[0].frame()

        for r in regions[1:]:
            assert r.frame() == frame, 'all regions have to belong to a single frame'

        inf = thresh_px * 2
        regions_yx = np.array([r.centroid() for r in regions])
        gt_yx = np.array(self.get_positions(frame))
        dist_mat = cdist(gt_yx, regions_yx)
        dist_mat[dist_mat > thresh_px] = inf
        matched_region_idx = np.argmin(dist_mat, axis=1)
        n_matches_for_regions = np.zeros(len(regions_yx), dtype=int)
        for idx in matched_region_idx:
            n_matches_for_regions[idx] += 1

        cardinalities = []
        for count in n_matches_for_regions:
            if count == 1:
                cardinalities.append('single')
            elif count > 1:
                cardinalities.append('multi')
            elif count == 0:
                cardinalities.append('noise')
            else:
                assert True
        return cardinalities

    def _get_index(self):
        return pd.MultiIndex.from_product([range(min(self.df.index.levels[0]), max(self.df.index.levels[0]) + 1),
                                           range(1, self.num_ids + 1)], names=['frame', 'id'])

    def fill_missing_positions_with_nans(self):
        self.df = self.df.reindex(index=self._get_index, fill_value=np.nan)
        return self.df

    def get_missing_positions(self):
        """
        Return frame and id pairs that are not defined in the ground truth.

        :return: DataFrame with frame and id columns
        """
        return self._get_index().to_frame()[self.x.isna()].reset_index(drop=True)

    def draw(self, frame_range=None, ids=None):
        import matplotlib.pylab as plt
        if frame_range is None:
            frame_range = (self.min_frame(), self.max_frame() + 1)
        if ids is None:
            ids = range(self.num_ids)
        yx = np.array([self.get_positions(frame) for frame in range(*frame_range)])  # shape=(frames, ids, yx)
        for i in ids:
            plt.plot(yx[:, i, 1], yx[:, i, 0], label=i)


class GtPermutationsMixin(object):
    def __init__(self):
        self.__permutation = {}
        self.__gt_id_to_real_permutation = {}
        for id_ in range(self.__num_ids):
            self.__gt_id_to_real_permutation[id_] = id_
            self.__permutation[id_] = id_

    def set_permutation_reversed(self, data):
        self.__permutation = self.get_permutation(data)
        temp = dict(self.__permutation)
        for key, val in temp.iteritems():
            self.__permutation[val] = key
            self.__gt_id_to_real_permutation[key] = val

    def set_permutation(self, data):
        """
        given list of tuples (frame, id, y, x)
        set internal permutation to fit given experiment

        Args:
            data:

        Returns:

        """

        self.__permutation = self.get_permutation(data)
        for key, val in self.__permutation.iteritems():
            self.__gt_id_to_real_permutation[val] = key

    def get_permutation_reversed(self):
        return self.__gt_id_to_real_permutation

    def get_permutation_dict(self):
        return self.__permutation

    def get_permutation(self, data):
        perm = {}
        for frame, id_, y, x in data:
            original_id_, _ = self.match_gt(frame, y, x)
            perm[id_] = original_id_

        return perm

    def permute(self, data):
        if isinstance(data, list):
            new_data = [None for _ in range(len(data))]
            for i, it in enumerate(data):
                new_data[self.__permutation[i]] = it

            return new_data
        elif isinstance(data, int):
            return self.__permutation[data]
        else:
            return None

from core.region.bbox import BBox


class GtDummyDetectorMixin(object):
    def init_gt_dummy_detector(self, bb_size_px, fp_prob=0, fn_prob=0,
                               true_detection_jitter_scale=None, false_detection_jitter_scale=None):
        self.detector_bb_size_px = bb_size_px
        self.detector_fp_prob = fp_prob  # probability of false positive detection per actual object in gt
        self.detector_fn_prob = fn_prob
        if true_detection_jitter_scale is None:
            self.true_scale = bb_size_px / 8
        else:
            self.true_scale = true_detection_jitter_scale
        if false_detection_jitter_scale is None:
            self.false_scale = bb_size_px
        else:
            self.false_scale = false_detection_jitter_scale

    def detect(self, frame_nr):
        bboxes = []
        bb_half_px = self.detector_bb_size_px / 2
        gt_detections = self.df.loc[frame_nr]
        for obj_id, det in gt_detections.iterrows():
            if np.random.rand() > self.detector_fn_prob:
                xy = [det.x, det.y] + np.random.normal(scale=self.true_scale, size=(2,))
                bbox = BBox(*np.concatenate((xy - bb_half_px, xy + bb_half_px)), frame=frame_nr)
                bbox.obj_id = obj_id
                bboxes.append(bbox)
            if np.random.rand() < self.detector_fp_prob:
                xy = [det.x, det.y] + np.random.normal(scale=self.false_scale, size=(2,))
                bbox = BBox(*np.concatenate((xy - bb_half_px, xy + bb_half_px)), frame=frame_nr)
                bbox.obj_id = None
                bboxes.append(bbox)
        return bboxes


class GtDummyReIdMixin(object):
    def init_gt_dummy_reid(self, match_beta_param=5, no_match_beta_param=5):
        self.reid_match_beta_param = match_beta_param
        self.reid_no_match_beta_param = no_match_beta_param

    def reid(self, bbox1, bbox2):
        if hasattr(bbox1, 'obj_id') and bbox1.obj_id is not None:
            match1 = bbox1.obj_id
        else:
            match1 = self.get_matching_obj_id(bbox1)
        if hasattr(bbox2, 'obj_id') and bbox2.obj_id is not None:
            match2 = bbox2.obj_id
        else:
            match2 = self.get_matching_obj_id(bbox2)
        if match1 is None or match2 is None or match1 != match2:
            # not matching
            return np.random.beta(1, self.reid_no_match_beta_param)
        else:
            # match
            return np.random.beta(self.reid_match_beta_param, 1)


if __name__ == '__main__':
    from core.project.project import Project
    import matplotlib.pylab as plt
    gt_filename = 'Cam1_clip.avi.'
    gt = GT('data/GT/' + gt_filename + 'txt')
    p = Project.from_dir('../projects/2_temp/Cam1_clip/190510_2038_fixed_seed_fixed_orientation')
    gt.set_project_offsets(p)

    # index = pd.MultiIndex.from_product([xrange(gt.min_frame(), gt.max_frame() + 1), xrange(1, gt.get_num_ids() + 1)],
    #                                    names=['frame', 'id'])
    # df = pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2', 'confidence'], index=index)
    #
    # frame_range = xrange(gt.min_frame(), gt.max_frame() + 1)
    # matches = gt.match_on_data(p, frame_range)
    # for frame, tracklet_ids in tqdm.tqdm(matches.items()):
    #     # img = p.img_manager.get_whole_img(frame)
    #     # plt.imshow(img)
    #     for gt_id, tracklet_id in enumerate(tracklet_ids, 1):
    #         if tracklet_id is not None:
    #             r = p.chm[tracklet_id].get_region_in_frame(frame)
    #             head, tail = r.get_head_tail()
    #             df.loc[frame, gt_id] = np.concatenate((head[::-1], tail[::-1], [1]))
    #     #     plt.plot(head[1], head[0], 'o')
    #     #     plt.plot(tail[1], tail[0], 'x')
    #     # plt.show()



    # class GtDetector(GtDummyDetectorMixin, GtDummyReIdMixin, GT):
    #     pass
    #
    # gt2 = GtDetector('data/GT/' + gt_filename + 'txt')
    # gt2.bbox_size_px = 70
    # gt2.init_gt_dummy_detector(gt2.bbox_size_px, fp_prob=0.05, fn_prob=0.001, false_detection_jitter_scale=40)
    # gt2.init_gt_dummy_reid()
    # # print(((gt.df[['x', 'y']] - gt2.df[['x', 'y']]) > 0.1).any(axis=1).nonzero())
    # # print(gt.df.type.value_counts(dropna=False))
    # detections = []
    # for frame in tqdm.tqdm(xrange(gt2.min_frame(), gt2.max_frame() + 1)):
    #     detections.append(gt2.detect(frame))
    #
    # pickle.dump(detections, open('detections.pkl', 'w'))
    # import pickle
    # dmax = 100
    # from collections import defaultdict
    # X = defaultdict(list)
    # y = defaultdict(list)
    # from itertools import product
    # for frame1 in tqdm.tqdm(xrange(gt2.min_frame(), gt2.max_frame() + 1)):
    #     for frame2 in range(frame1, min(frame1 + dmax, gt2.max_frame())):
    #         delta = frame2 - frame1
    #         for bb1, bb2 in product(detections[frame1], detections[frame2]):
    #             # spatial = ...
    #             if bb1.obj_id is None or bb2.obj_id is None or bb1.obj_id != bb2.obj_id:
    #                 y[delta].append(0)
    #             else:
    #                 y[delta].append(1)
    #             X[delta].append([bb1 - bb2, gt2.reid(bb1, bb2)])
    # pickle.dump({'X': X, 'y': y}, open('pairwise_features.pkl', 'w'))

#     import cv2
#     from core.project.project import Project
#     p = Project('/home/matej/prace/ferda/projects/2_temp/190131_1415_Cam1_ILP_cardinality_dense_fixed_orientation_json/')
#     from utils.video_manager import get_auto_video_manager
#     # vm = get_auto_video_manager(p)
#     for frame in tqdm.tqdm(range(100)):
#         img = p.img_manager.get_whole_img(frame)
#         for bbox in gt2.detect(frame):
#             bbox.move(-np.array([p.video_crop_model['x1'], p.video_crop_model['y1']])).draw_to_image(img)
#         cv2.imwrite('out/gtdetector/{:03d}.png'.format(frame), img)
#

for data in pos.data_vars.itervalues():
    data.values[data.values == -1] = np.nan