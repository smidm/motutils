from gt import GT
from utils.clearmetrics.clearmetrics import ClearMetrics
import numpy as np


class Evaluator:
    def __init__(self, config, gt):
        self.__config = config
        self.__gt = gt
        self.__clearmetrics = None


    def evaluate_FERDA(self, project, frame_limits_start=0, frame_limits_end=-1):
        from core.project.export import ferda_trajectories_dict
        print "PREPARING trajectories"
        single_trajectories = ferda_trajectories_dict(project, frame_limits_start=frame_limits_start,
                                                      frame_limits_end=frame_limits_end)

        # TODO: gt. set permutation
        self.evaluate(single_trajectories, frame_limits_start=0, frame_limits_end=frame_limits_end)


    def evaluate(self, data, frame_limits_start=0, frame_limits_end=-1):
        """
        data should be in the form as clearmetrics define,
        data = {frame1: [val1, val2, val3],
                frame2: [val1, None, val3, val4]
                frame3: [val1, val2]
               }
        Args:
            data:

        Returns:
        """

        # TODO: load from config
        dist_threshold = 30
        print "Preparing GT"
        gt = self.__gt.for_clearmetrics(frame_limits_start=frame_limits_start, frame_limits_end=frame_limits_end)
        print "evaluating"
        self.__clearmetrics = ClearMetrics(gt, data, dist_threshold)
        self.__clearmetrics.match_sequence()
        self.print_stats()

    # and others, will be called from comparatos
    def get_FP(self):
        pass

    def print_stats(self, float_precission=3):
        mismatches = self.__clearmetrics.get_mismatches_count()

        print "_____________________________"
        print "|--Clearmetrics statistics--|"
        print ("| MOTA: \t\t{:."+str(float_precission)+"%}").format(self.__clearmetrics.get_mota())
        print ("| MOTP: \t\t{:."+str(float_precission)+"}").format(self.__clearmetrics.get_motp())
        print ("| #FN: \t\t\t{:}").format(self.__clearmetrics.get_fn_count())
        print ("| #FP: \t\t\t{:}").format(self.__clearmetrics.get_fp_count())
        print ("| #mismatches: \t{:}").format(mismatches)
        print ("| #objects: \t{:}").format(self.__clearmetrics.get_object_count())
        print ("| #matches: \t{:}").format(self.__clearmetrics.get_matches_count())
        print "| "
        print "| legend: "
        print "| \tMOTA - Accuracy"
        print "| \tMOTP - precission"
        print "| \tFN: - lost"
        print "| \tmismatches - num of id swaps"
        print "| \t\t e.g."
        print "| \t\t\t _____ ...."
        print "| \t\t\t .....X____"
        print "| \t\t\t generates 2 mismatches"
        print "|___________________________|"

    def evaluate_idtracker(self, file, frame_limits_start=0, frame_limits_end=-1):
        import scipy.io as sio
        data = sio.loadmat(file)

        measurements = {}
        for frame, items in enumerate(data['trajectories']):
            measurements[frame] = []
            for it in items:
                if it is None:
                    measurements[frame].append(None)
                else:
                    # id tracker output definition
                    y = it[1]
                    x = it[0]
                    measurements[frame].append(np.array((y, x)))

        # TODO: do it better for videosequences where there is an interaction in first frame
        frame = 0
        permutation_data = []
        for id_, it in enumerate(measurements[frame]):
            permutation_data.append((frame, id_, it[0], it[1]))

        self.__gt.set_permutation(permutation_data)

        self.evaluate(measurements, frame_limits_start=0, frame_limits_end=frame_limits_end)

        self.detect_idswap(measurements)

    def detect_idswap(self, measurements):
        m_i = 0
        for frame in measurements:
            data = self.__gt.permute(measurements[frame])
            for id_, it in enumerate(data):
                y, x = it[0], it[1]
                match_id_, _ = self.__gt.match_gt(frame, y, x, limit_distance=30)
                id_ = self.__gt.permute(id_)
                if match_id_ is not None and id_ != match_id_:
                    m_i += 1
                    print "MISMATCH: #{:}, frame: {:}, id: {:}, gt_id: {:}".format(m_i, frame, id_, match_id_)


if __name__ == '__main__':
    from core.project.project import Project

    # p = Project()
    # p.load('/Users/flipajs/Documents/wd/zebrafish')
    #
    gt = GT()
    # gt.load(p.GT_file)
    gt.load('/Users/flipajs/Documents/dev/ferda/data/GT/5Zebrafish_nocover_22min.pkl')

    ev = Evaluator(None, gt)
    # ev.evaluate_FERDA(p, frame_limits_end=4498)
    # ev.evaluate_FERDA(p, frame_limits_end=14998)
    ev.evaluate_idtracker('/Volumes/Seagate Expansion Drive/FERDA-data/idTracker-5Zebrafish/trajectories.mat')
    # ev.evaluate_idtracker('/Users/flipajs/Dropbox/FERDA/idTracker_Cam1/trajectories_nogaps.mat')
