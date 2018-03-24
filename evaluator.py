from gt import GT
from utils.clearmetrics.clearmetrics import ClearMetrics
import numpy as np


class Evaluator:
    def __init__(self, config, gt):
        self.__config = config
        self._gt = gt
        self.__clearmetrics = None

    def eval_ids(self, project, frames=None, max_d=5, match=None):
        # print "evaluation in progress..."
        if match is None:
            match = self._gt.match_on_data(project, frames=frames, max_d=max_d, match_on='tracklets', permute=True)
            print "match done..."

        t_id_map = {}

        max_f = 0
        for frame, it in match.iteritems():
            max_f = max(max_f, frame)
            for id_, t_id in enumerate(it):
                if t_id not in t_id_map:
                    t_id_map[t_id] = set()

                t_id_map[t_id].add(id_)

        # for t_id, s in t_id_map.iteritems():
        #     t_id_map[t_id] = list(s)

        print "t_id_map DONE..."

        single_gt_len = 0
        single_len = 0
        mistakes_len = 0
        num_mistakes = 0

        mistakes = {}
        for t in project.chm.chunk_gen():
            t.id()

            gts = set()
            if t.id() in t_id_map:
                gts = t_id_map[t.id()]

            le = t.length()

            if len(gts) == 1:
                single_gt_len += le

                if t.P == gts:
                    single_len += le
                else:
                    num_mistakes += 1
                    mistakes_len += le
                    mistakes[t.id()] = gts


                    # if len(t.P) == 0 and len(gts) == 1:
                    #     num_mistakes += 1
                    #     mistakes_len += le
                    #     mistakes[t.id()] = gts
                    #
                    # elif len(t.P) == 1:
                    #     if t.P == gts:
                    #         single_len += le
                    #     else:
                    #         num_mistakes += 1
                    #         mistakes_len += le
                    #
                    #         mistakes[t.id()] = gts

        print "total correct coverage: {:.2%}".format(single_len / float(len(project.animals) * max_f))
        print "single correct coverage: {:.2%}".format(single_len / float(single_gt_len))
        print "total mistakes coverage: {:.2%}".format(mistakes_len / float(len(project.animals) * max_f))
        print "single mistakes coverage: {:.2%}".format(mistakes_len / float(single_gt_len))

    # def eval_ids_from_match(self, project, match, perm, frames=None, max_d=5, verbose=0):
    #     # print "evaluation in progress..."
    #
    #     t_id_map = {}
    #
    #     max_f = max(match.iterkeys())
    #
    #     # max_f = 0
    #     # for frame, it in match.iteritems():
    #     #     max_f = max(max_f, frame)
    #     #     for id_, t_id in enumerate(it):
    #     #         if t_id not in t_id_map:
    #     #             t_id_map[t_id] = set()
    #     #
    #     #         t_id_map[t_id].add(id_)
    #
    #     single_gt_len = 0
    #     single_len = 0
    #     mistakes_len = 0
    #     num_mistakes = 0
    #
    #     mistakes = []
    #
    #     gts = perm
    #     frame = 0
    #     for it in match.itervalues():
    #         not_m = False
    #         for val, i in enumerate(it):
    #             gt_val = -1
    #             if i is not None:
    #                 gt_val = gts[i]
    #
    #             single_gt_len += 1
    #
    #             if val == gt_val:
    #                 single_len += 1
    #             elif gt_val != -1:
    #                 num_mistakes += 1
    #                 mistakes_len += 1
    #
    #                 mistakes.append((frame, gt_val))
    #             else:
    #                 not_m = True
    #
    #         if not_m and verbose > 0:
    #             print frame, perm, it
    #
    #         frame += 1
    #
    #     print "Mistakes: ", mistakes
    #
    #     c_coverage = single_len/float(len(project.animals)*max_f)
    #     m_coverage = mistakes_len / float(len(project.animals) * max_f)
    #     # print "correct pose: {:.2%}".format(c_coverage)
    #     # # print "single correct coverage: {:.2%} ({})".format(single_len/float(single_gt_len), single_len)
    #     # print "wrong pose: {:.2%}".format(m_coverage)
    #     # print "unknown pose: {:.2%}".format(1-(m_coverage+c_coverage))
    #     # # print "single mistakes coverage: {:.2%}".format(mistakes_len/float(single_gt_len))
    #
    #     return c_coverage, m_coverage, single_len, mistakes_len

    def eval_ids_from_match(self, project, match, perm, frames=None, max_d=5, verbose=0):
        # print "evaluation in progress..."
        max_f = max(match.iterkeys())

        single_gt_len = 0
        single_len = 0
        mistakes_len = 0
        num_mistakes = 0

        mistaken_tracklets = set()
        mistakes = []

        gts = perm
        frame = 0
        for it in match.itervalues():
            not_m = False
            for i, tracklet_id in enumerate(it):
                val = None
                try:
                    val = list(project.chm[tracklet_id].P)[0]
                except:
                    pass

                gt_val = gts[i]

                single_gt_len += 1

                if val == gt_val:
                    single_len += 1
                elif val is not None:
                    num_mistakes += 1
                    mistakes_len += 1

                    mistakes.append((frame, gt_val, tracklet_id))
                    mistaken_tracklets.add(project.chm[tracklet_id])
                else:
                    not_m = True

            if not_m and verbose > 0:
                print frame, perm, it

            frame += 1

        print "Mistakes: ", mistakes
        print "Mistaken tracklets (#{}): ".format(len(mistaken_tracklets))
        for t in mistaken_tracklets:
            print t

        c_coverage = single_len / float(len(project.animals) * max_f)
        m_coverage = mistakes_len / float(len(project.animals) * max_f)
        # print "correct pose: {:.2%}".format(c_coverage)
        # # print "single correct coverage: {:.2%} ({})".format(single_len/float(single_gt_len), single_len)
        # print "wrong pose: {:.2%}".format(m_coverage)
        # print "unknown pose: {:.2%}".format(1-(m_coverage+c_coverage))
        # # print "single mistakes coverage: {:.2%}".format(mistakes_len/float(single_gt_len))

        return c_coverage, m_coverage, single_len, mistakes_len

    def evaluate_FERDA(self, project, frame_limits_start=0, frame_limits_end=-1, permutation_frame=0, step=1):
        from core.project.export import ferda_trajectories_dict
        from core.graph.region_chunk import RegionChunk
        print "PREPARING trajectories"
        single_trajectories = ferda_trajectories_dict(project, frame_limits_start=frame_limits_start,
                                                      frame_limits_end=frame_limits_end, step=step)

        # TODO: gt. set permutation
        # permutation_data = []
        # for t in project.chm.chunks_in_frame(permutation_frame):
        #     id_ = list(t.P)[0]
        #     rch = RegionChunk(t, project.gm, project.rm)
        #     c = rch.centroid_in_t(permutation_frame)
        #     permutation_data.append((permutation_frame, id_, c[0], c[1]))
        #
        # self.__gt.set_permutation(permutation_data)

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
        gt = self._gt.for_clearmetrics(frame_limits_start=frame_limits_start, frame_limits_end=frame_limits_end)
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
        print ("| MOTA: \t\t{:." + str(float_precission) + "%}").format(self.__clearmetrics.get_mota())
        print ("| MOTP: \t\t{:." + str(float_precission) + "}").format(self.__clearmetrics.get_motp())
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

        self._gt.set_permutation(permutation_data)

        self.evaluate(measurements, frame_limits_start=0, frame_limits_end=frame_limits_end)

        self.detect_idswap(measurements)

    def detect_idswap(self, measurements):
        m_i = 0
        for frame in measurements:
            data = self._gt.permute(measurements[frame])
            for id_, it in enumerate(data):
                y, x = it[0], it[1]
                match_id_, _ = self._gt.match_gt(frame, y, x, limit_distance=30)
                id_ = self._gt.permute(id_)
                if match_id_ is not None and id_ != match_id_:
                    m_i += 1
                    print "MISMATCH: #{:}, frame: {:}, id: {:}, gt_id: {:}".format(m_i, frame, id_, match_id_)


def draw_id_t_img(p, matches, perms, name=None, col_w=1, gt_h=5, gt_border=1, row_border=2, row_h=30, bg=[0, 0, 0],
                  impath=None):
    from core.animal import colors_
    import cv2

    num_trackers = len(matches)
    num_frames = len(matches[0])
    if num_trackers == 2:
        num_frames = min(len(matches[0]), len(matches[1]))
    num_objects = len(p.animals)

    im = np.zeros((row_h * num_objects * num_trackers, col_w * num_frames, 3), dtype=np.uint8)
    im[:, :, :] = bg

    for frame in range(num_frames):
        for id_ in range(num_objects):
            for tr in range(num_trackers):
                val = matches[tr][frame][id_]

                if val >= num_objects:
                    print frame

                y = id_ * row_h * num_trackers + tr * row_h
                if val is not None:
                    color = colors_[perms[tr][val]]
                    im[y:y + row_h, frame * col_w:(frame + 1) * col_w, :] = color

    for id_ in range(num_objects):
        # for tr in range(num_trackers):
        for i in range(num_objects):
            if perms[0][i] == id_:
                # gt_color = colors_[perms[0][i]]
                gt_color = colors_[id_]

        if num_trackers == 2:
            y = id_ * row_h * num_trackers + row_h / 2
        else:
            y = id_ * row_h * num_trackers

        yy = y + ((row_h - gt_h) / 2)
        yy2 = yy + gt_h
        # black margin
        im[yy - gt_border:yy, :, :] = [0, 0, 0]
        im[yy2:yy2 + gt_border, :, :] = [0, 0, 0]
        im[yy:yy2, :, :] = gt_color

    for id_ in range(1, num_objects):
        y = id_ * row_h * num_trackers
        im[y - row_border:y + row_border, :, :] = [0, 0, 0]

    # cv2.putText(im, 'idTracker', (7, 42), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 0), 2)
    # cv2.putText(im, 'FERDA', (7, 95), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 0), 2)
    # cv2.putText(im, 'idTracker', cv2.FONT, 0.2, (255, 255, 255), )
    # import cv2
    # cv2.imwrite(p.working_directory+'/temp/im.png', im)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    print im.shape
    plt.imshow(im)
    ax = plt.gca()

    h_ = num_trackers * row_h
    fsize = 6
    ax.set_yticks((np.arange(num_objects) * h_) + h_ / 2 - 2)
    # starting from 1
    ax.set_yticklabels(range(1, num_objects + 1), fontsize=fsize)

    xt = np.array(range(0, num_frames, 1000))
    ax.set_xticks(xt)
    ax.set_xticklabels(xt, fontsize=fsize)
    # plt.xlabel('frame', fontsize=fsize)
    # plt.ylabel('id', fontsize=fsize)

    if name is not None:
        plt.title(name, fontsize=fsize + 2)

    if impath is None:
        impath = p.working_directory + '/temp/overall_comparison.png'

    plt.savefig(impath, bbox_inches='tight', pad_inches=0, dpi=512)
    fig.tight_layout()

    plt.show()


def eval_centroids(p, gt, match=None):
    from tqdm import tqdm
    data = []
    for frame in tqdm(range(p.gm.end_t)):
        data.append(np.array([[np.nan, np.nan] for _ in range(len(p.animals))]))
        for t in p.chm.chunks_in_frame(frame):
            if len(t.P) == 1:
                id_ = list(t.P)[0]

                if id_ >= len(p.animals):
                    import warnings
                    warnings.warn("id_ > num animals t_id: {} id: {}".format(t.id(), id_))
                    continue

                c = p.rm[t.r_id_in_t(frame, p.gm)].centroid()

                data[frame][id_][0] = c[0]
                data[frame][id_][1] = c[1]

        data[frame] = np.array(data[frame])

    data = np.array(data)

    if match is None:
        # TODO: max_d is an important parameter. Document it and put it into GUI!
        # match = gt.match_on_data(p, data_centroids=data, match_on='centroids', max_d=25, frames=range(len(data)))
        match = gt.match_on_data(p, match_on='tracklets', max_d=5, frames=range(len(data)))

    # freq = np.zeros((len(p.animals), len(p.animals)), dtype=np.int)
    # for it in match.itervalues():
    #     for i, val in enumerate(it):
    #         freq[i][val] += 1
    #
    # m_ = np.argmax(freq, axis=0)
    # perm = {}
    #
    # for i in range(len(p.animals)):
    #     perm[i] = m_[i]

    perm = gt.get_permutation_dict()

    ev = Evaluator(None, gt)
    f_c_coverage, f_m_coverage, single_len, mistakes_len = ev.eval_ids_from_match(p, match, perm)

    return match, perm, f_c_coverage, f_m_coverage, single_len, mistakes_len


def print_coverage(c_coverage, m_coverage, singles_len='undef', mistakes_len='undef'):
    print "correct pose: {:.2%} (#{} frames)".format(c_coverage, singles_len)
    print "wrong pose: {:.2%} (#{} frames)".format(m_coverage, mistakes_len)
    print "unknown pose: {:.2%}".format(1 - (c_coverage + m_coverage))


def compare_trackers(p, idtracker_path=None, impath=None, name=None, skip_idtracker=False,
                     gt_ferda_perm=None, gt=None, draw=True):
    from utils.idtracker import load_idtracker_data

    if gt is None:
        gt = GT()
        gt.load(p.GT_file)

    if not skip_idtracker:
        data, _ = load_idtracker_data(idtracker_path, p, gt)
        data[:, :, 0], data[:, :, 1] = data[:, :, 1].copy(), data[:, :, 0].copy()

        # idTracker
        match = gt.match_on_data(p, data_centroids=data, match_on='centroids', max_d=25, frames=range(len(data)))
        freq = np.zeros((len(p.animals), len(p.animals)), dtype=np.int)
        for it in match.itervalues():
            for i, val in enumerate(it):
                freq[i][val] += 1

        m_ = np.argmax(freq, axis=0)
        perm = {}

        for i in range(len(p.animals)):
            perm[i] = m_[i]

    idtracker_m_coverage, idtracker_c_coverage = -1, -1
    if not skip_idtracker:
        print "IdTracker:"
        ev = Evaluator(None, gt)
        idtracker_c_coverage, idtracker_m_coverage, singles_len, mistakes_len = ev.eval_ids_from_match(p, match, perm)

        print_coverage(idtracker_c_coverage, idtracker_m_coverage)

    print "FERDA:"
    match2, perm2, f_c_coverage, f_m_coverage, singles_len, mistakes_len = eval_centroids(p, gt)
    print_coverage(f_c_coverage, f_m_coverage, singles_len=singles_len, mistakes_len=mistakes_len)

    if draw:
        if not skip_idtracker:
            draw_id_t_img(p, [match, match2], [perm, perm2], name=name, row_h=50, gt_h=10, gt_border=2,
                          bg=[200, 200, 200], impath=impath)
        else:
            if gt_ferda_perm is not None:
                perm2 = gt_ferda_perm
            draw_id_t_img(p, [match2], [perm2], name=name, row_h=50, gt_h=10, gt_border=2, bg=[200, 200, 200],
                          impath=impath)

    return (idtracker_c_coverage, idtracker_m_coverage, f_c_coverage, f_m_coverage)

def gt_find_permutation(project, gt, frame=None):
    from core.graph.region_chunk import RegionChunk
    # if get_separated_frame_callback:
    #     frame = self.get_separated_frame_callback()

    permutation_data = []
    for t in project.chm.chunks_in_frame(frame):
        if not t.is_single():
            continue

        id_ = list(t.P)[0]
        y, x = RegionChunk(t, project.gm, project.rm).centroid_in_t(frame)
        permutation_data.append((frame, id_, y, x))

    gt.set_permutation_reversed(permutation_data)

def evaluate_project(project_path, gt_path):
    from core.project.project import Project
    from utils.gt.gt import GT

    project = Project()
    project.load(project_path)

    # TODO: remove in future when an update is not necessary...
    project.chm.update_N_sets(project)

    gt = GT()
    gt.load(gt_path)

    gt.set_offset(y=project.video_crop_model['y1'],
                  x=project.video_crop_model['x1'],
                  frames=project.video_start_t
                  )

    # TODO: find best frame...
    gt_find_permutation(project, gt, frame=0)

    compare_trackers(project, skip_idtracker=True, gt_ferda_perm=gt.get_permutation_reversed(),
                     gt=gt, draw=False)


if __name__ == '__main__':
    evaluate_project('/Users/flipajs/Documents/wd/FERDA/Cam1', '/Users/flipajs/Documents/dev/ferda/data/GT/Cam1_.pkl')

    # from core.project.project import Project
    #
    # # p = Project()
    # # p.load('/Users/flipajs/Documents/wd/zebrafish')
    # #
    # gt = GT()
    # # gt.load(p.GT_file)
    # gt.load('/Users/flipajs/Documents/dev/ferda/data/GT/5Zebrafish_nocover_22min.pkl')
    #
    # ev = Evaluator(None, gt)
    # # ev.evaluate_FERDA(p, frame_limits_end=4498)
    # # ev.evaluate_FERDA(p, frame_limits_end=14998)
    # ev.evaluate_idtracker('/Volumes/Seagate Expansion Drive/FERDA-data/idTracker-5Zebrafish/trajectories.mat')
    # # ev.evaluate_idtracker('/Users/flipajs/Dropbox/FERDA/idTracker_Cam1/trajectories_nogaps.mat')
