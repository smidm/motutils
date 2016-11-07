from gt import GT
from utils.clearmetrics.clearmetrics.clearmetrics import ClearMetrics
import numpy as np


class Evaluator:
    def __init__(self, config, gt):
        self.__config = config
        self.__gt = gt
        self.__clearmetrics = None


    def evaluate_FERDA(self, project):
        from core.project.export import ferda_single_trajectories_dict
        print "PREPARING trajectories"
        single_trajectories = ferda_single_trajectories_dict(project, frame_limits_end=30)

        # TODO: gt. set permutation
        single_trajectories[1][1] = None
        self.evaluate(single_trajectories)


    def evaluate(self, data):
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
        gt = self.__gt.for_clearmetrics(frame_limits_end=30)
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
        print "|___________________________|"


if __name__ == '__main__':
    from core.project.project import Project

    p = Project()
    p.load('/Users/flipajs/Documents/wd/FERDA/Cam1_')

    gt = GT()
    gt.load(p.GT_file)

    ev = Evaluator(None, gt)
    ev.evaluate_FERDA(p)
