import cPickle as pickle
import sys
import warnings


class GT:
    """
    None means not defined

    self.__positions[frame][id] in format (y, x, type)

    format (y, x, type)

    type =  1 - clear, precise
            2..N - impreciese, inside collision, number signifies the num of ants in collision, it is also segmentation dependent...

    """
    def __init__(self, num_ids=0, num_frames=0, version=1.0):
        self.__num_ids = num_ids

        self.__positions = {}
        self.__behavior = {}

        self.__precision = 0
        self.__gt_version = version

        self.__min_frame = 0
        self.__max_frame = sys.maxint

    def get_num_ids(self):
        return self.__num_ids

    def load(self, path):
        try:
            with open(path, 'rb') as f:
                self._gt = pickle.load(f)

            print "GT was sucessfully loaded from ", path
        except:
            print "GT was not loaded ", path

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
            for i, (y, x, type) in enumerate(self.__positions[frame]):
                if type == 1:
                    p[i] = (y, x)

        return p

    def get_position(self, frame, id):
        return self.get_clear_positions(frame)[id]

    def set_position(self, frame, id, y, x, type=1):
        self.__set_frame(self.__positions, frame)
        self.__positions[frame][id] = (y, x, type)

    def save(self):
        # TODO: rename previous with name_BACKUP_DATE
        # TODO: save
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

        num_animals = len(project.animals)

        i = 0
        l = len(project.chm)
        print_progress(i, l, prefix='Progress:', suffix='Complete', barLength=50)

        for frame in range(frame_limits_start, frame_limits_end):
            self.__positions[frame] = [None for i in range(len(project.animals))]

        for t in project.chm.chunk_gen():
            print_progress(i, l, prefix='Progress:', suffix='Complete', barLength=50)
            i += 1

            if len(t.P.intersection(t.N)):
                warnings.warn("PN intersection is not empty! tracklet: "+str(t)+" P: "+str(t.P)+" N:"+str(t.N))
            # is decided
            elif len(t.P.union(t.N)) == num_animals:
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
                        for id_ in list(t.P):
                            self.__positions[frame][id_] = (r.centroid()[0], r.centroid()[1], len(t.P))

        print


    def match_gt(self, frame, position):
        # TODO:
        pass

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


if __name__ == '__main__':
    from core.project.project import Project
    p = Project()
    p.load('/Users/flipajs/Documents/wd/zebrafish')

    gt = GT()
    gt.build_from_PN(p)