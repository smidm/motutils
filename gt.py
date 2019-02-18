from __future__ import print_function
import pickle
import sys
import warnings
import numpy as np
import tqdm
from scipy.spatial.distance import cdist
import pandas as pd


class GT:
    """
    Ground truth handling.

    When working with a FERDA project, don't forget to set spatial and temporal offsets, see set_project_offsets().

    None means not defined.
    """
    def __init__(self, num_ids=0, version=1.0, precision=None):
        self.__num_ids = num_ids

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
        self.df = None
        """
        legacy ground truth storage:
        __positions[frame][id] in format (y, x, type)
    
        type =  1 - clear, precise
                2..N - impreciese, inside interaction, number signifies the num of ants in interaction, 
                it is also segmentation dependent...        
        """
        self.__positions = {}

        self.__precision = precision
        self.__gt_version = version

        self.__min_frame = 0
        self.__max_frame = sys.maxsize

        self.__permutation = {}
        self.__gt_id_to_real_permutation = {}
        self.__init_permutations()

        # gt (__positions) offset relative to the original video
        """
        if working in roi y: 100, x: 200, and starting in frame 100
        __gt_x_offset = 100
        """
        self.__gt_x_offset = 0
        self.__gt_y_offset = 0
        self.__gt_frames_offset = 0

        # working offset, set by inspected project relative to original video, see set_offset()
        self.__x_offset = 0
        self.__y_offset = 0
        self.__frames_offset = 0

        self.break_on_inconsistency = False

    @classmethod
    def from_mot(cls, filename):
        """
        Load Multiple Object Tacking Challenge trajectories file.

        :param filename: mot filename
        :return: DataFrame, columns frame and id start with 1 (MATLAB indexing)
        """
        df = pd.read_csv(filename, names=[u'frame', u'id', u'x', u'y', u'width', u'height', u'confidence'],
                         index_col=[u'frame'])
        # return df[(df.x != -1) & (df.y != -1)]
        df.index -= 1  # MATLAB to 0-based indexing
        del df['confidence']
        del df['width']
        del df['height']
        df['type'] = 1
        df = df[['id', 'y', 'x', 'type']]
        positions = {}
        for frame, df_frame in df.groupby(level=0):
            del df_frame['id']
            positions[frame] = df_frame.values

        gt = cls(df.id.nunique())
        gt.__positions = positions
        gt.__min_frame = df.index.min()
        gt.__max_frame = df.index.max()

        df = pd.read_csv(filename, names=[u'frame', u'id', u'x', u'y', u'width', u'height', u'confidence'],
                         index_col=[u'frame', u'id'], converters={u'frame': lambda x: int(x) - 1})
        gt.df = df[(df.x != -1) & (df.y != -1)]
        return gt

    def get_roi(self):
        """
        Return GT rectangular bounds.

        :return: xmin, xmax, ymin, ymax
        """
        yx = np.vstack([self.get_positions(frame) for frame in range(self.min_frame(), self.max_frame())])  # shape=(n, 2), [y, x]
        return yx[:, 1].min(), yx[:, 1].max(), yx[:, 0].min(), yx[:, 0].max()

    def __init_permutations(self):
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

    def get_num_ids(self):
        return self.__num_ids

    def load(self, path):
        # try:
        with open(path, 'rb') as f:
            tmp_dict = pickle.load(f)

        self.__dict__.update(tmp_dict)
        self.__init_permutations()

        print("GT was sucessfully loaded from " + path)
        # except:
        #     print "GT was not loaded ", path

    def get_all_ids_around(self, frame, position, max_distance=-1):
        if max_distance < 0:
            max_distance = self.__precision

        # TODO: based on __precision returns all ids in radius ordered by distance

        pass

    def set_project_offsets(self, project):
        self.set_offset(
            project.video_crop_model['x1'],
            project.video_crop_model['y1'],
            project.video_start_t)

    def set_offset(self, x=0, y=0, frames=0):
        """
        Set offset of an external project that will be applied to GT positions before operations with external values.
        """
        self.__x_offset = self.__gt_x_offset - x
        self.__y_offset = self.__gt_y_offset - y
        self.__frames_offset = self.__gt_frames_offset - frames

    def __set_frame(self, d, frame):
        if frame not in d:
            d[frame] = [None for _ in range(self.__num_ids)]

    def get_clear_positions(self, frame):
        frame -= self.__frames_offset
        p = [None for _ in range(self.__num_ids)]
        if frame in self.__positions:
            for i, it in enumerate(self.__positions[frame]):
                if it is not None:
                    y, x, type_ = it
                    if type_ == 1:
                        p[self.__permutation[i]] = (y + self.__y_offset, x + self.__x_offset)

        return p

    def get_positions(self, frame):
        frame -= self.__frames_offset
        p = [None for _ in range(self.__num_ids)]
        if frame in self.__positions:
            for i, it in enumerate(self.__positions[frame]):
                if it is not None:
                    y, x, _ = it
                    p[self.__permutation[i]] = (y + self.__y_offset, x + self.__x_offset)

        return p

    def get_clear_positions_dict(self):
        positions = {}
        for f in self.__positions.iterkeys():
            positions[f] = self.get_clear_positions(f)

        return positions

    def get_position(self, frame, id_):
        id_ = self.__permutation[id_]
        return self.get_clear_positions(frame)[id_]

    def set_position(self, frame, id_, y, x, type_=1):
        frame -= self.__frames_offset
        self.__set_frame(self.__positions, frame)
        id_ = self.__permutation[id_]
        self.__positions[frame][id_] = (y - self.__y_offset, x - self.__x_offset, type_)

    def save(self, path, make_copy=True):
        import os
        import datetime

        if make_copy:
            if os.path.exists(path):
                dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                os.rename(path, path[:-4]+'_'+dt+'.pkl')

        with open(path, 'w') as f:
            pickle.dump(self.__dict__, f, -1)

        pass

    def build_from_PN(self, project, frame_limits_start=0, frame_limits_end=-1):
        """
        for each tracklet get info from P and N sets
        Returns:

        """
        from utils.misc import print_progress
        from core.graph.region_chunk import RegionChunk
        print("... CREATING GT from PN sets ...")

        if frame_limits_end < 0:
            from utils.video_manager import get_auto_video_manager
            v = get_auto_video_manager(project)
            frame_limits_end  = v.total_frame_count()

        self.__min_frame = frame_limits_start
        self.__max_frame = frame_limits_end
        self.__num_ids = len(project.animals)

        num_animals = len(project.animals)

        i = 0
        l = len(project.chm)
        print_progress(i, l, prefix='Progress:', suffix='Complete', barLength=50)

        for frame in range(frame_limits_start, frame_limits_end):
            self.__positions[frame] = [None for i in range(self.__num_ids)]

        for t in project.chm.chunk_gen():
            print_progress(i, l, prefix='Progress:', suffix='Complete', barLength=50)
            i += 1

            if len(t.P.intersection(t.N)):
                warnings.warn("PN intersection is not empty! tracklet: "+str(t)+" P: "+str(t.P)+" N:"+str(t.N))
            # is decided
            # elif len(t.P.union(t.N)) == num_animals:
            else:
                rch = RegionChunk(t, project.gm, project.rm)
                for r in rch.regions_gen():
                    frame = r.frame()

                    if frame_limits_start > frame:
                        continue

                    if frame_limits_end <= frame:
                        break

                    if len(t.P) == 1:
                        id_ = list(t.P)[0]
                        self.__positions[frame][id_] = (r.centroid()[0], r.centroid()[1], 1)
                    else:
                        for id_ in list(set(range(num_animals)) - t.N):
                            self.__positions[frame][id_] = (r.centroid()[0], r.centroid()[1], len(t.P))

        self.__init_permutations()
        print()

    def match_gt(self, frame, y, x, limit_distance=None):
        """
        if self.__precision is not None use this as max distance for match

        return
        Args:
            frame:
            y:
            x:
            limit_distance: float >= 0 if provided it overrides self.__precision

        Returns:
            id, (y, x)
            None, None if no match
        """
        p = self.get_clear_positions(frame)

        best_dist = np.inf
        if self.__precision is not None:
            best_dist = self.__precision
        if limit_distance is not None:
            best_dist = limit_distance

        best_id = None
        for id_, data in enumerate(p):
            if data is None:
                continue
            else:
                y_, x_ = data

            d = ((y-y_)**2 + (x-x_)**2)**0.5
            if d < best_dist:
                best_dist = d
                best_id = id_

        if best_id is None:
            return None, (-1, -1)
        else:
            best_id_ = best_id
            # for key, val in self.__permutation.iteritems():
            #     if best_id == val:
            #         best_id_ = key

        return best_id_, (p[best_id][0], p[best_id][1])

    def import_from_txt(self):
        # TODO:
        pass

    def export2txt(self):
        # TODO:
        pass

    def export2cvs(self):
        # TODO:
        pass

    def export2pkl(self):
        # TODO:
        pass

    def export2json(self):
        # TODO:
        pass

    def for_clearmetrics(self, frame_limits_start=0, frame_limits_end=-1):
        gt = {}

        if frame_limits_end < 0:
            frame_limits_end = self.__max_frame

        for frame in range(frame_limits_start, frame_limits_end):
            gt[frame] = []
            for it in self.get_positions(frame):
                if it is not None:
                    it = np.array(it)

                gt[frame].append(it)

        return gt

    def min_frame(self):
        return self.__min_frame

    def max_frame(self):
        """

        :return: max frame number + 1
        """
        return self.__max_frame

    def check_none_occurence(self):
        print("Checking None occurence")
        for frame, vals in self.__positions.iteritems():
            for it in vals:
                if it is None:
                    print(frame)

                # Old way of representing undefined
                if it[0] < 50 and it[1] < 100:
                    print("UNDEF POS:", frame)

        print("DONE")

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
            frames = range(self.min_frame(), self.max_frame())

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

    def test_edge(self, tracklet1, tracklet2, project):
        return self.tracklet_id_set(tracklet1, project) == self.tracklet_id_set(tracklet2, project)

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

    def draw(self, frame_range=None, ids=None):
        import matplotlib.pylab as plt
        if frame_range is None:
            frame_range = (self.min_frame(), self.max_frame() + 1)
        if ids is None:
            ids = range(self.get_num_ids())
        yx = np.array([gt.get_positions(frame) for frame in range(*frame_range)])  # shape=(frames, ids, yx)
        for i in ids:
            plt.plot(yx[:, i, 1], yx[:, i, 0], label=i)


if __name__ == '__main__':
    gt = GT()
    gt.load('data/GT/Cam1_clip.avi.pkl')

    gt2 = GT.from_mot('data/GT/Cam1_clip.avi.txt')
    sys.exit(0)
    from core.project.project import Project
    p = Project()

    name = 'Camera3'
    # name = 'Sowbug3'
    name = 'Cam1'
    # name = 'zebrafish'
    nogaps = ''
    # nogaps = '_nogaps'
    playground = ''
    playground = '_playground'

    # wd = '/Users/flipajs/Documents/wd/FERDA/Cam1_playground'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Zebrafish_playground'
    wd = '/Users/flipajs/Documents/wd/FERDA/'+name+playground
    wd = '/Users/flipajs/Documents/wd/FERDA/Cam1_rf'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Sowbug3'

    p.load_semistate(wd, 'id_classified')

    p.chm.add_single_vertices_chunks(p)
    p.gm.update_nodes_in_t_refs()

    # p.GT_file = '/Users/flipajs/Documents/dev/ferda/data/GT/5Zebrafish_nocover_22min.pkl'
    # p.save()

    gt = GT()
    gt.load(p.GT_file)

    path = '/Users/flipajs/Documents/wd/idTracker/'+name+'/trajectories'+nogaps+'.mat'
    from evaluator import compare_trackers

    results = compare_trackers(p, path, impath='/Users/flipajs/Documents/dev/ferda/thesis/results/overall_'+name+nogaps+'_rf.png')

    # gt.project_stats(p)

    if False:
        epsilons = []
        edges = []
        variant = []
        AA = []
        BB = []

        theta = 0.5

        out_t = 0
        test_t = 0


        import time

        for v in p.gm.active_v_gen():
            t = time.time()
            e, es = p.gm.get_2_best_out_edges_appearance_motion_mix(v)
            out_t += time.time() - t

            if e[1] is not None:
                A = es[0]
                B = es[1]
                t = time.time()
                if gt.test_edge(p.gm.get_chunk(e[0].source()), p.gm.get_chunk(e[0].target()), p):
                    eps = (A / theta) - (A + B)
                    variant.append(0)
                else:
                    eps = (A + B) / ((1/theta) - 1)
                    variant.append(1)

                test_t += time.time() - t

                AA.append(A)
                BB.append(B)
                epsilons.append(eps)
                edges.append((int(e[0].source()), int(e[0].target())))

        print(min(epsilons), max(epsilons))

        print("T: ", out_t, test_t)

        with open(p.working_directory+'/temp/epsilons.pkl', 'wb') as f:
            pickle.dump((epsilons, edges, variant, AA, BB), f)
