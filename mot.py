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
import warnings
import scipy.optimize


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
    df = pd.read_csv(filename, names=[u'frame', u'id', u'x', u'y', u'width', u'height', u'confidence'],
                     index_col=[u'frame', u'id'])
    return df[(df.x != -1) & (df.y != -1)]


def visualize_mot(video_file, out_video_file, df_mots, names=None,
                  montage_max_wh=(1920, 1200), duration=None):
    from moviepy.video.tools.drawing import blit
    from moviepy.video.io.VideoFileClip import VideoFileClip
    from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip, clips_array
    from moviepy.video.VideoClip import ColorClip
    from moviepy.video.fx.resize import resize
    from itertools import count
    import cv2  # TODO: remove dependency

    def generate_colors(count):
        import matplotlib.pylab as plt
        cm = plt.get_cmap('gist_rainbow')
        return np.array([cm(1. * i / count, bytes=True)[:3] for i in range(count)]).astype(float)

    def generate_markers(n, colors):
        """
        Generate circular bitmap markers with alpha masks.
        """
        from moviepy.video.tools.drawing import circle
        radius = 10
        blur = radius * 0.2
        img_dim = radius * 2 + 1
        img_size = (img_dim, img_dim)
        pos = (radius, radius)
        markers = []
        for c in colors:  # random_colors:
            img = circle(img_size, pos, radius, c, blur=blur)
            mask = circle(img_size, pos, radius, 1, blur=blur)
            markers.append({'img': img, 'mask': mask})
        return markers, pos

    def process_image(get_frame_fun, t, df, name, markers, marker_pos, colors, id_to_gt, fps):
        """
        Draw single tracker data on a frame.

        :param img:
        :param df:
        :param name:
        :param markers:
        :param marker_pos:
        :param counter:
        :param id_to_gt: list or dict, maps df ids to ground truth ids, used to display same markers for all results
        :return:
        """
        if id_to_gt is None:
            id_to_gt = range(len(markers))  # identity mapping
        img = get_frame_fun(t).copy()
        frame = int(round((t * fps)))
        if frame + 1 in df.index.levels[0]:
            for obj_id, row in df.loc[frame + 1].iterrows():  # mot data in df are indexed from 1
                if not (np.isnan(row.x) or np.isnan(row.y) or row.x == -1 or row.y == -1):
                    if not ('width' in row and 'height' in row) or np.isnan(row.width) or row.width == -1:
                        marker = markers[id_to_gt[obj_id]]
                        img = blit(marker['img'], img, (int(row.x) - marker_pos[0], int(row.y) - marker_pos[1]),
                             mask=marker['mask'])
                    else:
                        cv2.arrowedLine(img, (int(row.x), int(row.y)), (int(row.x + row.width), int(row.y + row.height)),
                                        colors[id_to_gt[obj_id]], thickness=2, line_type=cv2.LINE_AA, tipLength=0.2)
        if name is not None:
            font_size = 1.5
            font_thickness = 2
            text_size, _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)
            cv2.putText(img, name, (int((img.shape[1] - text_size[0]) / 2), 60),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), font_thickness)
        cv2.putText(img, str(frame), (30, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        return img

    def limit_size(wh, max_wh):
        """
        Limit video clip size.

        :param wh: actual width and height
        :param max_wh: maximal width and height
        :return: newsize suitable for moviepy.video.fx.all.resize
        """
        wh = np.array(wh, dtype=float)
        size_ratio = wh / max_wh
        if np.count_nonzero(size_ratio > 1) == 0:
            newsize = None  # keep original size
        else:
            # rescale according to most exceeding dimension to fit into max_wh
            newsize = 1. / size_ratio.max()
            assert np.all(wh * newsize <= max_wh)
        return newsize

    def make_fun(df, name, markers, marker_pos=None, rgb_cycle=None, id_to_gt=None, fps=None):
        return lambda gf, t: process_image(gf, t, df, name, markers, marker_pos, rgb_cycle, id_to_gt, fps)

    def detections_to_array(df, frame=None):
        """
        Get xy detections as an array.

        :param df: mot trajectories DataFrame
        :param frame: if None use frame with most ids defined
        :return: xy values, trajectory ids, frame number
        """
        if frame is None:
            frame = df.groupby('frame').size().idxmax()
        df_xy = df.loc[frame, ['x', 'y']]
        return df_xy.values, df_xy.index, frame

    def find_mapping(df, df_ref):
        """
        Find spatially close mapping to reference trajectory ids.

        Ids that are not matched are mapped to itself.

        :param df: trajectories
        :param df_ref: reference trajectories
        :return: dict, df ids to reference ids
        """
        # get positions in a suitable frame
        positions, ids, frame = detections_to_array(df)
        ref_positions, ref_ids, _ = detections_to_array(df_ref, frame=frame)
        assert len(ref_ids) == len(df_ref.index.get_level_values(1).unique()), \
            'all identities have to be defined in ground truth at the selected frame'
        # match positions
        distance_matrix = np.vectorize(lambda i, j: np.linalg.norm(positions[i] - ref_positions[j])) \
            (*np.indices((len(ids), len(ref_ids))))
        ii, jj = scipy.optimize.linear_sum_assignment(distance_matrix)
        if np.count_nonzero(distance_matrix[ii, jj] > 10):
            warnings.warn('large distance beween detection and gt ' + str(distance_matrix[ii, jj]))
        id_to_gt = dict(zip(ids, jj))
        # add identity mappings for ids not present in the selected frame
        all_ids = df.index.get_level_values(1).unique()
        if len(ids) < len(all_ids):
            ids_without_gt_match = set(all_ids) - set(ids)
            id_to_gt.update(zip(ids_without_gt_match, ids_without_gt_match))  # add identity mapping
        return id_to_gt

    MONTAGE_GRID_WH = [(0, 0), (1, 1), (2, 1), (3, 1), (2, 2), (3, 2), (3, 2)]  # montage grid sizes for 0-6 number of images
    n_clips = len(df_mots)
    clip = VideoFileClip(video_file)
    n_colors = max([df.index.get_level_values('id').max() for df in df_mots])  # max number of identities
    rgb_cycle = generate_colors(n_colors)
    markers, marker_pos = generate_markers(n_colors, rgb_cycle)

    if names is not None:
        assert len(names) == len(df_mots)
        if 'gt' in names:
            reference_idx = names.index('gt')
        elif 'ground truth' in names:
            reference_idx = names.index('ground truth')
        else:
            reference_idx = 0
    else:
        reference_idx = 0
    mappings = [find_mapping(df, df_mots[reference_idx]) for df in df_mots]

    clips = []
    for i, df in enumerate(df_mots):
        if names is not None:
            name = names[i]
        else:
            name = None
        # text_clip = TextClip(name, size=(200, 100), color='white').set_position('center', 'top')  # , fontsize=100
        video_clip = clip.fl(make_fun(df, name, markers, marker_pos=marker_pos, rgb_cycle=rgb_cycle,
                                      id_to_gt=mappings[i], fps=clip.fps))
        # clips.append(CompositeVideoClip([video_clip, text_clip], use_bgclip=True)) # , bg_color='red')) # , use_bgclip=True)) , size=clip.size
        clips.append(video_clip)

    n_montage_cells = MONTAGE_GRID_WH[n_clips][0] * MONTAGE_GRID_WH[n_clips][1]
    for _ in range(n_montage_cells - n_clips):
        clips.append(ColorClip(clips[-1].size, (0, 0, 0), duration=clips[-1].duration))
    out_clip = clips_array(np.array(clips).reshape((MONTAGE_GRID_WH[n_clips][1], -1)))

    newsize = limit_size(out_clip.size, montage_max_wh)
    if newsize is not None:
        out_clip = out_clip.fx(resize, newsize)

    if duration is not None:
        out_clip = out_clip.set_duration(duration)

    out_clip.write_videofile(out_video_file)  # , threads=4


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
        visualize_mot(args.video_in, args.video_out, dfs, args.input_names)  #, duration=3)


