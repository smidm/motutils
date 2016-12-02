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

    def match_on_data(self, project, max_d=50):
        from scipy.spatial.distance import cdist
        from utils.misc import print_progress
        from itertools import izip

        print "Matching..."

        num_frames = self.max_frame() - self.min_frame()

        not_matched = []
        match = {}
        i = 0
        for frame in range(self.min_frame(), self.max_frame()):
            match[frame] = [None for _ in range(len(project.animals))]

            # add chunk ids
            r_t = project.gm.regions_and_t_ids_in_t(frame)
            regions = [x[0] for x in r_t]
            ch_ids = [x[1] for x in r_t]

            centroids = np.array([r.centroid() for r in regions])
            pos = self.__positions[frame]
            pos = np.array([(x[0], x[1]) for x in pos])

            dists = cdist(pos, centroids)
            m1_i = np.argmin(dists, axis=1)
            m1 = dists[range(pos.shape[0]), m1_i]

            for a_id, id_ in enumerate(m1_i):
                if m1[a_id] > max_d:
                    # try if inside region...
                    for r, t_id in izip(regions, ch_ids):
                        if r.is_inside(pos[a_id], tolerance=5):
                            match[frame][a_id] = t_id
                            break

                    if match[frame][a_id] is None:
                        not_matched.append(frame)
                else:
                    match[frame][a_id] = ch_ids[id_]

            # TODO: solve big distances for oversegmented regions
            # dists[range(pos.shape[0]), m1_i] = np.inf
            # m2 = np.min(dists, axis=1)

            i += 1

            if i % 10 == 0:
                print_progress(i, num_frames)

        print "Done.."
        print "Not matched in frames ", not_matched



        return match

if __name__ == '__main__':
    from core.project.project import Project
    p = Project()
    # p.load('/Users/flipajs/Documents/wd/FERDA/Cam1_')
    p.load('/Users/flipajs/Documents/wd/zebrafish')
    p.GT_file = '/Users/flipajs/Documents/dev/ferda/data/GT/5Zebrafish_nocover_22min.pkl'
    p.save()

    gt = GT()
    gt.build_from_PN(p)
    gt.save('/Users/flipajs/Documents/dev/ferda/data/GT/5Zebrafish_nocover_22min.pkl')

    # gt = GT()
    # gt.build_from_PN(p)
    #
    # # gt.load('/Users/flipajs/Documents/dev/ferda/data/GT/5Zebrafish_nocover_22min.pkl')
    # gt.save('/Users/flipajs/Documents/dev/ferda/data/GT/Cam1_.pkl')
    #
    # print gt.get_clear_positions(100)
    # print gt.get_clear_rois(100)