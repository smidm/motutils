"""
Multi object tracking results and ground truth

- conversion,
- evaluation,
- visualization.

For more help run this file as a script with --help parameter.
"""
import pandas as pd
import errno
import numpy as np
import sys

assert sys.version_info >= (3, 5)


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
    df_out['width'] = -1
    df_out['height'] = -1
    df_out['confidence'] = -1
    return df_out


def load_toxtrack(filename, topleft_xy=(0, 0)):
    """
    Load ToxTrack results.

    Example Tracking_0.txt:
    0	0	1	194.513	576.447	1
    1	0	1	192.738	580.313	1
    2	0	1	190.818	584.126	1
    3	0	1	188.84	588.213	1
    4	0	1	186.78	592.463	1

    :param filename: idTracker results (Tracking_0.txt)
    :param topleft_xy: tuple, length 2; xy coordinates of the arena top left corner
    :return: DataFrame with frame 	id 	x 	y 	width 	height 	confidence columns
    """
    df = pd.read_csv(filename, delim_whitespace=True,
                     names=['frame', '_0', 'id', 'x', 'y', '_1'],
                     usecols=['frame', 'id', 'x', 'y'])
    df['frame'] += 1  # MATLAB indexing
    df['x'] += topleft_xy[0]
    df['y'] += topleft_xy[1]
    df = df.assign(width=-1)
    df = df.assign(height=-1)
    df = df.assign(confidence=-1)
    df.sort_values(['frame', 'id'], inplace=True)
    return df


def save_mot(filename, df):
    df.to_csv(filename, header=False, index=False)


def load_mot(filename):
    return pd.read_csv(filename, names=[u'frame', u'id', u'x', u'y', u'width', u'height', u'confidence'])


def get_matplotlib_color_cycle():
    import matplotlib
    import matplotlib.pylab as plt
    matplotlib.colors.colorConverter.to_rgb

    rgb_cycle = []
    for hex_color in plt.rcParams['axes.prop_cycle'].by_key()['color']:
        rgb_cycle.append([x * 255 for x in matplotlib.colors.colorConverter.to_rgb(hex_color)])
    return rgb_cycle


def visualize_mot(video_file, out_video_file, df_mot):
    import cv2
    import tqdm
    # import math

    cap = cv2.VideoCapture(video_file)
    ret, img = cap.read()
    if not ret:
        raise IOError(errno.ENOENT, video_file + ' not found')
    img_shape = img.shape
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap = cv2.VideoCapture(video_file)

    writer = cv2.VideoWriter(out_video_file,
                             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),  # ('X', '2', '6', '4'),
                             cap.get(cv2.CAP_PROP_FPS),  # int(math.ceil(cap.get(cv2.CAP_PROP_FPS))),
                             img_shape[:2][::-1])

    rgb_cycle = get_matplotlib_color_cycle()
    for frame in tqdm.tqdm(range(1, frame_count + 1)):  # 100)): #
        ret, img = cap.read()
        assert ret
        for _, row in df_mot[df_mot.frame == frame].iterrows():
            if not (np.isnan(row.x) or np.isnan(row.y)):
                cv2.circle(img, (int(row.x), int(row.y)), 8, rgb_cycle[int(row.id - 1)], -1)
        cv2.putText(img, str(frame), (30, 30), cv2.FONT_HERSHEY_PLAIN, 0.65, (255, 255, 255))
        writer.write(img)
    writer.release()


def eval_mot(df_gt, df_results, sqdistth=10000):
    from motmetrics.utils import compare_to_groundtruth
    import motmetrics as mm
    columns_mapper = {'frame': 'FrameId', 'id': 'Id'}
    acc = compare_to_groundtruth(df_gt.set_index(['frame', 'id']).rename(columns=columns_mapper),
                                 df_results.set_index(['frame', 'id']).rename(columns=columns_mapper),
                                 dist='euc', distfields=['x', 'y'], distth=sqdistth)
    mh = mm.metrics.create()
    return mh.compute(acc, metrics=mm.metrics.motchallenge_metrics)


if __name__ == '__main__':
    import tqdm
    import argparse

    parser = argparse.ArgumentParser(description='Convert and visualize mot ground truth and results.')
    parser.add_argument('--load-tox', type=str, help='load ToxTracker trajectories (e.g., Tracking_0.txt)')
    parser.add_argument('--tox-topleft-xy', nargs='+', type=int, help='position of the arena top left corner, see first tuple in the Arena line in Stats_1.txt')
    parser.add_argument('--load-idtracker', type=str, help='load IdTracker trajectories (e.g., trajectories.txt)')
    parser.add_argument('--load-mot', type=str, help='load a MOT challenge csv file')
    parser.add_argument('--load-gt', type=str, help='load ground truth from a MOT challenge csv file')
    parser.add_argument('--video-in', type=str, help='input video file')
    parser.add_argument('--video-out', type=str, help='write visualization to a video file')
    parser.add_argument('--write-mot', type=str, help='write trajectories to a MOT challenge csv file')
    parser.add_argument('--eval', action='store_true', help='evaluate results')
    parser.add_argument('--write-eval', type=str, help='write evaluation results as a csv file')
    args = parser.parse_args()

    if args.load_tox:
        assert args.tox_topleft_xy, 'specify position of the arena top left corner using --tox-topleft-xy'
        assert len(args.tox_topleft_xy), 'need to pass exactly two values with --tox-topleft-xy'
        df = load_toxtrack(args.load_tox, topleft_xy=args.tox_topleft_xy)
    elif args.load_idtracker:
        df = load_idtracker(args.load_idtracker)
    elif args.load_mot:
        df = load_mot(args.load_mot)
    else:
        assert False, 'no input files specified'

    if args.write_mot:
        save_mot(args.write_mot, df)

    if args.video_out:
        assert args.video_in
        visualize_mot(args.video_in, args.video_out, df)

    if args.eval or args.write_eval:
        assert args.load_gt
        df_gt = load_mot(args.load_gt)
        print('Evaluating...')
        summary = eval_mot(df_gt, df)
        import motmetrics as mm
        mh = mm.metrics.create()
        print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
        if args.write_eval:
            summary.to_csv(args.write_eval, index=False)
