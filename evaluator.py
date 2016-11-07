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
        dist_threshold = 10
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
        print "_____________________________"
        print "|--Clearmetrics statistics--|"
        print ("| MOTA: \t\t{:."+str(float_precission)+"%}").format(self.__clearmetrics.get_mota())
        print ("| MOTP: \t\t{:."+str(float_precission)+"}").format(self.__clearmetrics.get_motp())
        print ("| #FN: \t\t\t{:}").format(self.__clearmetrics.get_fn_count())
        print ("| #FP: \t\t\t{:}").format(self.__clearmetrics.get_fp_count())
        print ("| #mismatches: \t{:}").format(self.__clearmetrics.get_mismatches_count())
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


if __name__ == '__main__':
    from core.project.project import Project

    p = Project()
    p.load('/Users/flipajs/Documents/wd/FERDA/Cam1_')

    gt = GT()
    gt.load(p.GT_file)

    ev = Evaluator(None, gt)
    ev.evaluate_FERDA(p, frame_limits_end=4498)
