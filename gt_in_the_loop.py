from core.id_detection.learning_process import LearningProcess
from core.project.project import Project
from utils.gt.gt import GT
import numpy as np


def test_one_id_in_tracklet(t, num_animals):
    return len(t.P) == 1 and \
           len(t.N) == num_animals- 1


def get_len_undecided(project, lp):
    coverage = 0
    max_ = 0
    for t in project.chm.chunk_gen():
        if t.id() not in lp.undecided_tracklets:
            continue

        if t.is_single():
            coverage += t.length()

        end_f_ = t.end_frame(project.gm)
        max_ = max(max_, end_f_)

    return coverage

def get_coverage(project, undecided=False, lp=None):
    coverage = 0
    max_ = 0
    for t in project.chm.chunk_gen():
        if test_one_id_in_tracklet(t, len(project.animals)):
            coverage += t.length()

        end_f_ = t.end_frame(project.gm)
        max_ = max(max_, end_f_)

    return coverage / float(max_ * len(project.animals))

def check_gt(p, tracklet_gt_map, step, already_reported):
    for t in p.chm.chunk_gen():
        if t.id() in already_reported:
            continue

        if t.id() in tracklet_gt_map:
            gt = tracklet_gt_map[t.id()]
        else:
            gt = set()

        t_n_fail = False
        for id_ in gt:
            if id_ in t.N:
                t_n_fail = True

        if (len(t.P) == 1 and t.P != gt) or t_n_fail:
            already_reported.add(t.id())

            print "STEP: {}, t_id: {}, t_len: {}, t.P: {}, t.N: {}, gt: {}".format(step, t.id(), t.length(), t.P, t.N, gt)
            try:
                print "\t decision cert: {}, tracklet measurements: {}".format(t.decision_cert, t.measurements)
            except:
                pass

def \
        assign_ids(p, semistate='tracklets_s_classified',
               features=['fm_idtracker_i.sqlite3', 'fm_basic.sqlite3'],
               gt_in_the_loop=False, out_state_name='id_classified',
               HIL=False, rf_max_features='auto',
               rf_min_new_samples_to_retrain=10000,
               rf_retrain_up_to_min=np.inf, auto_init_method='max_sum', num_runs='1',
               check_lp_steps=False):

    p.load_semistate(wd, state='tracklets_s_classified', one_vertex_chunk=True, update_t_nodes=True)

    gt = GT()
    gt.load(p.GT_file)

    lp = LearningProcess(p, verbose=1)
    lp.min_new_samples_to_retrain = rf_min_new_samples_to_retrain
    lp.rf_retrain_up_to_min = rf_retrain_up_to_min
    lp.map_decisions = check_lp_steps

    # lp.load_features('fm_basic.sqlite3')
    lp.load_features(features)

    # best_frame = lp.auto_init()
    best_frame = lp.auto_init(method=auto_init_method)
    permutation_data = []
    for d in lp.user_decisions:
        t = p.chm[d['tracklet_id_set']]
        id_ = d['ids'][0]
        y, x = RegionChunk(t, p.gm, p.rm).centroid_in_t(best_frame)
        permutation_data.append((best_frame, id_, y, x))

    gt.set_permutation(permutation_data)

    match = gt.match_on_data(p)
    tracklet_gt_map = {}

    perm = gt.get_permutation_reversed()
    for frame, vals in match.iteritems():
        for a_id, t_id in enumerate(vals):
            if t_id not in tracklet_gt_map:
                tracklet_gt_map[t_id] = set()
            else:
                tracklet_gt_map[t_id].add(perm[a_id])

    print len(lp.user_decisions)

    # IMPORTANT!
    if HIL:
        lp.ignore_inconsistency = False
        lp.set_eps_certainty(.5)
    else:
        lp.ignore_inconsistency = True
        # TODO: fix naming...
        # in fact it is 1-0.8 ...
        lp.set_eps_certainty(1.0)

    results = []
    for i in range(num_runs):
        increase_init_set = 0
        finished = True

        already_reported = set()

        init_training_set = None
        for run in range(100):
            print "---------_ RUN #{} _---------".format(run)
            tset = lp.reset_learning()
            if not init_training_set:
                init_training_set = tset

            step = 0
            while True:
                step += 1
                lp.next_step()

                if check_lp_steps:
                    check_gt(p, tracklet_gt_map, step, already_reported)

                if lp.consistency_violated or run < increase_init_set:
                    if run < increase_init_set:
                        t_id = lp.question_to_increase_smallest(gt).id()
                        # t_id = lp.get_best_question().id()
                    else:
                        t_id = lp.last_id
                        user_d_ids = [it['tracklet_id_set'] for it in lp.user_decisions]
                        if t_id in user_d_ids or t_id < 1:
                            t_id = lp.get_best_question().id()

                    t = p.chm[t_id]
                    t_class, animal_id = gt.get_class_and_id(t, p)
                    t.segmentation_class = t_class

                    # TODO: if allowed, remove if...
                    # mutli ID decision not allowed yet...
                    if len(animal_id) == 1:
                        lp.user_decisions.append({'tracklet_id_set': t_id, 'type': 'P', 'ids': animal_id})

                    finished = False

                    print "/// "
                    print lp.user_decisions
                    print "////"
                    print "User input. T id: {}, aid: {} class: {}".format(t_id, animal_id, t_class)
                    print "BREAKING... {} tracklets left undecided (sum len: {}). User decisions: {}. Coverage: {:.2%}".format(
                        len(lp.undecided_tracklets), get_len_undecided(p, lp), len(lp.user_decisions), get_coverage(p))
                    break
                elif len(lp.tracklet_certainty) == 0 and run >= increase_init_set:
                    finished = True

                    print "FINISHED"
                    break

            if finished:
                break



        from utils.gt.evaluator import eval_centroids, print_coverage

        print "RESULTS"
        _, _, cc, mc = eval_centroids(p, gt)
        print_coverage(cc, mc)

        results.append({'cc': cc, 'mc': mc, 'tset': init_training_set, 'HIL': run})

        if lp.ignore_inconsistency:
            p.save_semistate(state=out_state_name+'_no_HIL'+'_'+str(i))
        else:
            p.save_semistate(state=out_state_name+'_'+str(i))

    return results

if __name__ == '__main__':
    from core.graph.region_chunk import RegionChunk
    from thesis.config import *
    import cPickle as pickle
    import datetime
    p = Project()
    wd = '/Users/flipajs/Documents/wd/FERDA/Cam1_playground'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Cam1_rf'
    # wd = '/Users/flipajs/Documents/wd/FERDA/zebrafish_playground'
    wd = '/Users/flipajs/Documents/wd/FERDA/Camera3'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Sowbug3'
    # p.load_semistate('/Users/flipajs/Documen ts/wd/FERDA/Sowbug3', state='eps_edge_filter')

    config = {'rf_max_features': '0.5', 'HIL': False,
              'features': [
                           'fm_idtracker_i.sqlite3',
                           'fm_idtracker_c.sqlite3',
                           'fm_basic.sqlite3',
                           'fm_colornames.sqlite3',
                           # 'fm_colornames_lvl1.sqlite3',
                           'fm_hog.sqlite3',
                           'fm_lbp.sqlite3',
                           ],
              'rf_min_new_samples_to_retrain': 10,
              'rf_retrain_up_to_min': 200,
              'auto_init_method': 'max_min',
              'num_runs': 5,
              'wd': wd,
              'check_lp_steps': True,
              'semistate': 'tracklets_s_classified2'}

    result = assign_ids(p,
               rf_max_features=config['rf_max_features'],
               HIL=config['HIL'],
               features=config['features'],
               rf_min_new_samples_to_retrain=config['rf_min_new_samples_to_retrain'],
               rf_retrain_up_to_min=config['rf_retrain_up_to_min'],
               auto_init_method=config['auto_init_method'], num_runs=config['num_runs'],
               check_lp_steps=config['check_lp_steps'],
               semistate=config['semistate'])

    dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(RESULT_WD+'/id_assignment/'+wd.split('/')[-1]+dt, 'wb') as f:
        pickle.dump((config, result), f)

    print
    print config
