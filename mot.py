"""
Multi object tracking results and ground truth

- conversion,
- evaluation,
- visualization.

For more help run this file as a script with --help parameter.

TODO: merge with utils.gt.gt
"""
import pandas as pd
import numpy as np
import sys
import warnings
import scipy.optimize
from utils.gt.visualize import visualize

metrics_higher_is_better = ['idf1', 'idp', 'idr', 'recall', 'precision','mota']
metrics_lower_is_better = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'motp', 'motp_px']


def load_idtracker(filename):
    """
    Load idTracker results.

    Example trajectories.txt:

    X1	Y1	ProbId1	X2	Y2	ProbId2	X3	Y3	ProbId3	X4	Y4	ProbId4	X5	Y5	ProbId5
    459.85	657.37	NaN	393.9	578.17	NaN	603.95	244.9	NaN	1567.3	142.51	NaN	371.6	120.74	NaN
    456.43	664.32	NaN	391.7	583.05	NaN	606.34	242.57	NaN	1565.3	138.53	NaN	360.93	121.86	NaN
    453.22	670.03	NaN	389.63	587.08	NaN	608.41	240.66	NaN	1566.8	132.25	NaN	355.92	122.81	NaN
    ...

    :param filename: idTracker results (trajectories.txt or trajectories_nogaps.txt)
    :return: DataFrame with frame 	id 	x 	y 	width 	height 	confidence columns
    """
    df = pd.read_csv(filename, delim_whitespace=True)
    df.index += 1
    n_animals = len(df.columns) // 3
    for i in range(1, n_animals + 1):
        df[i] = i
    df['frame'] = df.index

    objs = []
    for i in range(1, n_animals + 1):
        objs.append(
            df[['frame', i, 'X' + str(i), 'Y' + str(i)]].rename({'X' + str(i): 'x', 'Y' + str(i): 'y', i: 'id'}, axis=1))
    df_out = pd.concat(objs)
    df_out.sort_values(['frame', 'id'], inplace=True)
    df[df.isna()] = -1
    df_out['width'] = -1
    df_out['height'] = -1
    df_out['confidence'] = -1
    return df_out


def load_toxtrac(filename, topleft_xy=(0, 0)):
    """
    Load ToxTrack results.

    Example Tracking_0.txt:
    0	0	1	194.513	576.447	1
    1	0	1	192.738	580.313	1
    2	0	1	190.818	584.126	1
    3	0	1	188.84	588.213	1
    4	0	1	186.78	592.463	1

    Documentation of the file format is in
    [ToxTrac: a fast and robust software for tracking organisms](https://arxiv.org/pdf/1706.02577.pdf) page 33.

    :param filename: Toxtrac results (Tracking_0.txt)
    :param topleft_xy: tuple, length 2; xy coordinates of the arena top left corner
    :return: DataFrame with frame 	id 	x 	y 	width 	height 	confidence columns
    """
    df = pd.read_csv(filename, delim_whitespace=True,
                     names=['frame', 'arena', 'id', 'x', 'y', 'label'],
                     usecols=['frame', 'id', 'x', 'y'])
    df['frame'] += 1  # MATLAB indexing
    df['x'] += topleft_xy[0]
    df['y'] += topleft_xy[1]
    df = df.assign(width=-1)
    df = df.assign(height=-1)
    df = df.assign(confidence=-1)
    df.sort_values(['frame', 'id'], inplace=True)
    df[df.isna()] = -1
    return df


def save_mot(filename, df):
    df.to_csv(filename, header=False, index=False)


def load_mot(filename):
    """
    Load Multiple Object Tacking Challenge trajectories file.

    :param filename: mot filename
    :return: DataFrame, columns frame and id start with 1 (MATLAB indexing)
    """
    df = pd.read_csv(filename, index_col=[u'frame', u'id'])  # names=[u'frame', u'id', u'x', u'y', u'width', u'height', u'confidence']
    return df[(df.x != -1) & (df.y != -1)]


def mot_in_roi(df, roi):
    """
    Limit MOT to a region of interest.

    :param df: MOT trajectories, DataFrame
    :param roi: utils.roi.ROI
    :return: MOT trajectories, DataFrame
    """
    idx_in_roi = (df.x >= roi.x()) & (df.y >= roi.y()) & (df.x < roi.x() + roi.width()) & (
                df.y < roi.y() + roi.height())
    return df[idx_in_roi]


def eval_mot(df_gt, df_results, sqdistth=10000):
    """
    Evaluate trajectories by comparing them to a ground truth.

    :param df_gt: ground truth DataFrame, columns <frame>, <id>, <x>, <y>; <frame> and <id> are 1-based; see load_mot
    :param df_results: result trajectories DataFrame, format same as df_gt
    :param sqdistth: square of the distance threshold, only detections and ground truth objects closer than
                     the threshold can be matched
    :return: (summary DataFrame, MOTAccumulator)
    """
    nan_mask = (df_results.x == -1) | (df_results.x == -1) | df_results.x.isna() | df_results.y.isna()
    if len(df_results[nan_mask]) > 0:
        warnings.warn('stripping nans from the evaluated trajectories')
        df_results = df_results[~nan_mask]
    from motmetrics.utils import compare_to_groundtruth
    import motmetrics as mm
    acc = compare_to_groundtruth(df_gt, df_results, dist='euc', distfields=['x', 'y'], distth=sqdistth)
    mh = mm.metrics.create()
    # remove id_global_assignment metric, workaround for https://github.com/cheind/py-motmetrics/issues/19
    metrics = mh.names[:]
    metrics.remove('id_global_assignment')
    return mh.compute(acc, metrics), acc  # metrics=mm.metrics.motchallenge_metrics


def eval_and_save(gt_file, mot_results_file, out_csv=None):
    """
    Evaluate results and save metrics.

    :param gt_file: ground truth filename (MOT format)
    :param mot_results_file: results filename (MOT format)
    :param out_csv: output file with a summary
    """
    df_gt = load_mot(gt_file)
    df_results = load_mot(mot_results_file)
    print('Evaluating...')
    assert sys.version_info >= (3, 5), 'motmetrics requires Python 3.5'
    summary, acc = eval_mot(df_gt, df_results)
    summary['motp_px'] = np.sqrt(summary['motp'])  # convert from square pixels to pixels
    import motmetrics as mm
    # mh = mm.metrics.create()
    print(mm.io.render_summary(summary))
    if out_csv is not None:
        summary.to_csv(out_csv, index=False)


def results_to_mot(results):
    """
    Create MOT challenge format DataFrame out of trajectories array.

    :param results: ndarray, shape=(n_frames, n_animals, 2 or 4); coordinates are in yx order, nan when id not present
    :return: DataFrame with frame, id, x, y, width, height and confidence columns
    """
    assert results.ndim == 3
    assert results.shape[2] == 2 or results.shape[2] == 4
    objs = []
    columns = ['x', 'y']
    indices = [1, 0]
    if results.shape[2] == 4:
        columns.extend(['width', 'height'])
        indices.extend([3, 2])
    for i in range(results.shape[1]):
        df = pd.DataFrame(results[:, i, indices], columns=columns)
        df['frame'] = range(1, results.shape[0] + 1)
        df = df[~(df.x.isna() | df.y.isna())]
        df['id'] = i + 1
        df = df[['frame', 'id'] + columns]
        objs.append(df)

    df = pd.concat(objs)

    df.sort_values(['frame', 'id'], inplace=True)
    if results.shape[2] == 2:
        df['width'] = -1
        df['height'] = -1
    df['confidence'] = -1
    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert and visualize mot ground truth and results.')
    parser.add_argument('--load-tox', type=str, help='load ToxTracker trajectories (e.g., Tracking_0.txt)')
    parser.add_argument('--tox-topleft-xy', nargs='+', type=int, help='position of the arena top left corner, see first tuple in the Arena line in Stats_1.txt')
    parser.add_argument('--load-idtracker', type=str, help='load IdTracker trajectories (e.g., trajectories.txt)')
    parser.add_argument('--load-mot', type=str, nargs='+', help='load a MOT challenge csv file(s)')
    parser.add_argument('--load-gt', type=str, help='load ground truth from a MOT challenge csv file')
    parser.add_argument('--video-in', type=str, help='input video file')
    parser.add_argument('--video-out', type=str, help='write visualization(s) to a video file')
    parser.add_argument('--write-mot', type=str, help='write trajectories to a MOT challenge csv file')
    parser.add_argument('--eval', action='store_true', help='evaluate results')
    parser.add_argument('--write-eval', type=str, help='write evaluation results as a csv file')
    parser.add_argument('--input-names', type=str, nargs='+', help='names of input MOT files')
    args = parser.parse_args()

    if args.load_tox:
        assert args.tox_topleft_xy, 'specify position of the arena top left corner using --tox-topleft-xy'
        assert len(args.tox_topleft_xy), 'need to pass exactly two values with --tox-topleft-xy'
        dfs = [load_toxtrac(args.load_tox, topleft_xy=args.tox_topleft_xy)]
    elif args.load_idtracker:
        dfs = [load_idtracker(args.load_idtracker)]
    elif args.load_mot:
        dfs = [load_mot(mot) for mot in args.load_mot]
    else:
        assert False, 'no input files specified'

    if args.write_mot:
        assert len(dfs) == 1
        save_mot(args.write_mot, dfs[0])

    if args.eval or args.write_eval:
        assert args.load_gt
        assert len(args.load_mot) == 1, 'only single input file can be specified for evaluation'
        eval_and_save(args.load_gt, args.load_mot[0], args.write_eval)

    if args.video_out:
        assert args.video_in
        visualize_mot(args.video_in, args.video_out, dfs, args.input_names)  # , duration=3)
