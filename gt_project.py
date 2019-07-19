import numpy as np
from scipy.spatial.distance import cdist
import tqdm


class GtProjectMixin(object):
    def __get_ids_from_match(self, match, t_id):
        return set([id_ for id_, x in enumerate(match) if x == t_id])

    def set_project_offsets(self, project):
        self.add_delta(
            -project.video_crop_model['x1'] if project.video_crop_model is not None else 0,
            -project.video_crop_model['y1'] if project.video_crop_model is not None else 0,
            -project.video_start_t)

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

    def test_tracklet_consistency(self, tracklet, match, ids=None):
        if ids is None:
            ids = self.__get_ids_from_match(match[0], tracklet.id())

        for i in range(1, len(match)):
            if ids != self.__get_ids_from_match(match[i], tracklet.id()):
                print("CONSISTENCY I, ", i)

                if self.break_on_inconsistency:
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

