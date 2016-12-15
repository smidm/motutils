import cPickle as pickle
import sys
import warnings
import numpy as np

class GT:
    """
    None means not defined

    self.__positions[frame][id] in format (y, x, type)

    self.__rois[frame][id] in format (y1, x1, y2, x2, type), where (y1, x1) is top left corner, (y2, x2) is bottom right.

    type =  1 - clear, precise
            2..N - impreciese, inside collision, number signifies the num of ants in collision, it is also segmentation dependent...

    """
    def __init__(self, num_ids=0, num_frames=0, version=1.0, precision=None):
        self.__num_ids = num_ids

        self.__positions = {}
        self.__rois = {}
        self.__behavior = {}

        self.__precision = precision
        self.__gt_version = version

        self.__min_frame = 0
        self.__max_frame = sys.maxint

        self.__permutation = {}

        self.__init_permutations()

    def __init_permutations(self):
        self.__permutation = {}
        for id_ in range(self.__num_ids):
            self.__permutation[id_] = id_


    def set_permutation(self, data):
        """
        given list of tuples (frame, id, y, x)
        set internal permutation to fit given experiment

        Args:
            data:

        Returns:

        """

        self.__permutation = self.get_permutation(data)

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

        print "GT was sucessfully loaded from ", path
        # except:
        #     print "GT was not loaded ", path

    def get_all_ids_around(self, frame, position, max_distance=-1):
        if max_distance < 0:
            max_distance = self.__precision

        # TODO: based on __precision returns all ids in radius ordered by distance

        pass

    def __set_frame(self, d, frame):
        if frame not in d:
            d[frame] = [None for _ in range(self.__num_ids)]

    def get_clear_positions(self, frame):
        p = [None for _ in range(self.__num_ids)]
        if frame in self.__positions:
            for i, it in enumerate(self.__positions[frame]):
                if it is not None:
                    y, x, type_ = it
                    if type_ == 1:
                        p[self.__permutation[i]] = (y, x)

        return p

    def get_positions(self, frame):
        p = [None for _ in range(self.__num_ids)]
        if frame in self.__positions:
            for i, it in enumerate(self.__positions[frame]):
                if it is not None:
                    y, x, _ = it
                    p[self.__permutation[i]] = (y, x)

        return p

    def get_clear_positions_dict(self):
        positions = {}
        for f in self.__positions.iterkeys():
            positions[f] = self.get_clear_positions(f)

        return positions

    def get_clear_rois(self, frame):
        p = [None for _ in range(self.__num_ids)]
        if frame in self.__rois:
            for i, it in enumerate(self.__rois[frame]):
                if it is not None:
                    y1, x1, y2, x2, type_  = it
                    if type_ == 1:
                        p[self.__permutation[i]] = (y1, x1, y2, x2)

        return p

    def get_position(self, frame, id_):
        id_ = self.__permutation[id_]
        return self.get_clear_positions(frame)[id_]

    def set_position(self, frame, id_, y, x, type_=1):
        self.__set_frame(self.__positions, frame)
        id_ = self.__permutation[id_]
        self.__positions[frame][id_] = (y, x, type_)

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
        print "... CREATING GT from PN sets ..."

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
            self.__rois[frame] = [None for i in range(self.__num_ids)]

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

                    roi = r.roi()

                    y1, x1 = roi.top_left_corner()
                    y2, x2 = roi.bottom_right_corner()

                    if frame_limits_start > frame:
                        continue

                    if frame_limits_end <= frame:
                        break

                    if len(t.P) == 1:
                        id_ = list(t.P)[0]
                        self.__positions[frame][id_] = (r.centroid()[0], r.centroid()[1], 1)
                        self.__rois[frame][id_] = (y1, x1, y2, x2, 1)
                    else:
                        for id_ in list(set(range(num_animals)) - t.N):
                            self.__positions[frame][id_] = (r.centroid()[0], r.centroid()[1], len(t.P))
                            self.__rois[frame][id_] = (y1, x1, y2, x2, len(t.P))

        self.__init_permutations()
        print

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
        return self.__max_frame

    def check_none_occurence(self):
        print "Checking None occurence"
        for frame, vals in self.__positions.iteritems():
            for it in vals:
                if it is None:
                    print frame

                # Old way of representing undefined
                if it[0] < 50 and it[1] < 100:
                    print "UNDEF POS:", frame

        print "DONE"

    def match_on_data(self, project, frames=None, max_d=5, match_on='tracklets'):
        from scipy.spatial.distance import cdist
        from utils.misc import print_progress
        from itertools import izip

        # num_frames = self.max_frame() - self.min_frame()

        not_matched = []
        match = {}
        i = 0

        if frames is None:
            frames = range(self.min_frame(), self.max_frame())

        for frame in frames:
            match[frame] = [None for _ in range(len(project.animals))]

            # add chunk ids
            if match_on=='tracklets':
                r_t = project.gm.regions_and_t_ids_in_t(frame)
                regions = [x[0] for x in r_t]
                ch_ids = [x[1] for x in r_t]
            else:
                regions = project.gm.regions_in_t(frame)

            if len(regions) == 0:
                continue

            centroids = np.array([r.centroid() for r in regions])
            pos = self.__positions[frame]
            if None in pos:
                continue
            pos = np.array([(x[0], x[1]) for x in pos])

            try:
                dists = cdist(pos, centroids)
            except:
                print centroids, regions, frame

            m1_i = np.argmin(dists, axis=1)
            m1 = dists[range(pos.shape[0]), m1_i]

            for a_id, id_ in enumerate(m1_i):
                if m1[a_id] > max_d:
                    # try if inside region...
                    if match_on == 'tracklets':
                        for r, t_id in izip(regions, ch_ids):
                            if r.is_inside(pos[a_id], tolerance=max_d):
                                if match_on == 'tracklets':
                                    match[frame][a_id] = t_id
                                break
                    else:
                        for r in regions:
                            if r.is_inside(pos[a_id], tolerance=max_d):
                                match[frame][a_id] = r.id()
                                break

                    if match[frame][a_id] is None:
                        not_matched.append(frame)
                else:
                    if match_on == 'tracklets':
                        match[frame][a_id] = ch_ids[id_]
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

    def tracklet_id_set(self, tracklet, project):
        """


        Args:
            tracklet:
            project:

        Returns:

        """
        match = self.match_on_data(project, range(tracklet.start_frame(project.gm), tracklet.end_frame(project.gm) + 1))

        keys = sorted([k for k in match.iterkeys()])
        match = [match[k] for k in keys]

        ids = self.__get_ids_from_match(match[0], tracklet.id())
        if self.test_tracklet_consistency(tracklet, match, ids):
            return [self.__permutation[id_] for id_ in ids]
        else:
            warnings.warn('Tracklet id: {} is inconsistent'.format(tracklet.id()))
            print match, ids

        return None

    def __get_ids_from_match(self, match, t_id):
        return set([id_ for id_, x in enumerate(match) if x == t_id])

    def test_tracklet_consistency(self, tracklet, match, ids=None):
        if ids is None:
            ids = self.__get_ids_from_match(match[0], tracklet.id())

        for i in range(1, len(match)):
            if ids != self.__get_ids_from_match(match[i], tracklet.id()):
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

        frame = tracklet.start_frame(project.gm) - 1
        if frame > 0:
            match = [x for x in self.match_on_data(project, [frame]).itervalues()][0]
            if self.__match_mapping_possible(match, ids):
                return False

        frame = tracklet.end_frame(project.gm) + 1
        if frame < self.__max_frame:
            match = [x for x in self.match_on_data(project, [frame]).itervalues()][0]
            if self.__match_mapping_possible(match, ids):
                return False

        return True

    def test_edge(self, tracklet1, tracklet2, project):
        return self.tracklet_id_set(tracklet1, project) == self.tracklet_id_set(tracklet2, project)

    def project_stats(self, p):
        num_max_len = 0
        not_consistent = 0

        not_consistent_list = []

        for t in p.chm.chunk_gen():
            match = [x for x in self.match_on_data(p, frames=range(t.start_frame(p.gm), t.end_frame(p.gm) + 1)).itervalues()]
            if not self.test_tracklet_consistency(t, match):
                not_consistent += 1
                not_consistent_list.append(t)

            if self.test_tracklet_max_len(t, p):
                num_max_len += 1

        num_t = len(p.chm)
        print "#max_len: {}, ({:.2%}, #not consitent: {})".format(num_max_len, num_max_len/float(num_t), not_consistent)

if __name__ == '__main__':
    from core.project.project import Project
    p = Project()
    # p.load('/Users/flipajs/Documents/wd/FERDA/Cam1_playground')
    p.load('/Users/flipajs/Documents/wd/FERDA/Zebrafish_playground')

    with open(p.working_directory+'/temp/isolation_score.pkl', 'rb') as f:
    # with open(wd+'/temp/isolation_score.pkl', 'rb') as f:
    # with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/isolation_score.pkl', 'rb') as f:
        up = pickle.Unpickler(f)
        p.gm.g = up.load()
        up.load()
        chm = up.load()
        p.chm = chm

    from core.region.region_manager import RegionManager
    p.rm = RegionManager(p.working_directory+'/temp', db_name='part0_rm.sqlite3')
    p.gm.rm = p.rm

    p.chm.add_single_vertices_chunks(p, frames=range(4500))
    p.gm.update_nodes_in_t_refs()


    # p.load('/Users/flipajs/Documents/wd/zebrafish')
    # p.GT_file = '/Users/flipajs/Documents/dev/ferda/data/GT/5Zebrafish_nocover_22min.pkl'
    # p.save()

    gt = GT()
    gt.load(p.GT_file)

    epsilons = []
    edges = []
    variant = []
    symmetric = []

    theta = 0.5

    for v in p.gm.active_v_gen():
        e, es = p.gm.get_2_best_out_edges_appearance_motion_mix(v)

        if e[1] is not None:
            # e_, es_ = p.gm.get_2_best_in_edges_appearance_motion_mix(e[0].source())
            # if e_[0].target() == e[0].target() or (e_[1] is not None and e_[1].target() == e[0].target()):
            #     symmetric.append(1)
            # else:
            #     symmetric.append(0)

            A = es[0]
            B = es[1]
            if gt.test_edge(p.gm.get_chunk(e[0].source()), p.gm.get_chunk(e[0].target()), p):
                eps = (A / theta) - (A + B)
                variant.append(0)
            else:
                eps = (A + B) / ((1/theta) - 1)
                variant.append(1)


            epsilons.append(eps)
            edges.append((int(e[0].source()), int(e[0].target())))

    print min(epsilons), max(epsilons)

    with open(p.working_directory+'/temp/epsilons', 'wb') as f:
        pickle.dump((epsilons, edges, variant), f)

    # gt.project_stats(p)

    # t1 = p.gm.get_chunk(762)
    # print gt.tracklet_id_set(t1, p)
    #
    # t2 = p.gm.get_chunk(784)
    # print gt.tracklet_id_set(t2, p)
    #
    # print gt.tracklet_id_set(p.gm.get_chunk(891), p)

    # gt.build_from_PN(p)
    # gt.save('/Users/flipajs/Documents/dev/ferda/data/GT/5Zebrafish_nocover_22min.pkl')

    # gt = GT()
    # gt.build_from_PN(p)
    #
    # # gt.load('/Users/flipajs/Documents/dev/ferda/data/GT/5Zebrafish_nocover_22min.pkl')
    # gt.save('/Users/flipajs/Documents/dev/ferda/data/GT/Cam1_.pkl')
    #
    # print gt.get_clear_positions(100)
    # print gt.get_clear_rois(100)