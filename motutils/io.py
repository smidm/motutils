"""
Multi object tracking results and ground truth

- conversion,
- evaluation,
- visualization.

For more help run this file as a script with --help parameter.

PyCharm debugger could have problems debugging inside this module due to a bug:
https://stackoverflow.com/questions/47988936/debug-properly-with-pycharm-module-named-io-py
workaround: rename the file temporarily

TODO: merge with utils.gt.gt
"""
import warnings

import numpy as np
import pandas as pd
import tqdm

from .bbox_mot import BboxMot
from .mot import Mot
from .posemot import PoseMot

metrics_higher_is_better = ["idf1", "idp", "idr", "recall", "precision", "mota"]
metrics_lower_is_better = [
    "num_false_positives",
    "num_misses",
    "num_switches",
    "num_fragmentations",
    "motp",
    "motp_px",
]


def load_any_mot(filename_or_buffer):
    df = pd.read_csv(filename_or_buffer, nrows=2)
    try:
        filename_or_buffer.seek(0)
    except AttributeError:
        pass
    try:
        for s in df.columns:
            float(s)
        bbox_mot = True
    except ValueError:
        bbox_mot = False
    if bbox_mot:
        mot = BboxMot(filename_or_buffer=filename_or_buffer)
    elif "keypoint" in df.columns:
        mot = PoseMot(filename_or_buffer=filename_or_buffer)
    else:
        mot = Mot(filename_or_buffer=filename_or_buffer)
    return mot


def load_idtracker(filename_or_buffer):
    """
    Load idTracker results.

    Example trajectories.txt:

    X1	Y1	ProbId1	X2	Y2	ProbId2	X3	Y3	ProbId3	X4	Y4	ProbId4	X5	Y5	ProbId5
    459.85	657.37	NaN	393.9	578.17	NaN	603.95	244.9	NaN	1567.3	142.51	NaN	371.6	120.74	NaN
    456.43	664.32	NaN	391.7	583.05	NaN	606.34	242.57	NaN	1565.3	138.53	NaN	360.93	121.86	NaN
    453.22	670.03	NaN	389.63	587.08	NaN	608.41	240.66	NaN	1566.8	132.25	NaN	355.92	122.81	NaN
    ...

    :param filename_or_buffer: idTracker results (trajectories.txt or trajectories_nogaps.txt)
    :return: DataFrame with frame 	id 	x 	y 	width 	height 	confidence columns
    """
    df = pd.read_csv(filename_or_buffer, delim_whitespace=True)
    df.index += 1
    n_animals = len(df.columns) // 3
    for i in range(1, n_animals + 1):
        df[i] = i
    df["frame"] = df.index

    objs = []
    for i in range(1, n_animals + 1):
        objs.append(
            df[["frame", i, "X" + str(i), "Y" + str(i)]].rename(
                {"X" + str(i): "x", "Y" + str(i): "y", i: "id"}, axis=1
            )
        )
    df_out = pd.concat(objs)
    df_out.sort_values(["frame", "id"], inplace=True)
    df[df.isna()] = -1
    df_out["width"] = -1
    df_out["height"] = -1
    df_out["confidence"] = -1
    return df_out


def load_idtrackerai(filename_or_buffer):
    """
    Load idtracker.ai results

    :param filename_or_buffer: idTracker results (trajectories.txt or trajectories_nogaps.txt)
    :return: DataFrame with frame 	id 	x 	y 	width 	height 	confidence columns
    """
    traj_ndarray = np.load(filename_or_buffer, allow_pickle=True)
    traj_dict = traj_ndarray.item()
    n_frames, n_ids, _ = traj_dict["trajectories"].shape

    frames = np.repeat(np.arange(1, n_frames + 1), n_ids).reshape(n_frames, n_ids, 1)
    obj_ids = np.tile(np.arange(1, n_ids + 1), n_frames).reshape(n_frames, n_ids, 1)
    df = pd.DataFrame(
        np.concatenate((frames, obj_ids, traj_dict["trajectories"]), axis=2).reshape(
            (n_frames * n_ids, 4)
        ),
        columns=["frame", "id", "x", "y"],
    )
    df = df.astype({"frame": "int", "id": "int"})
    df[df.isna()] = -1
    df["width"] = -1
    df["height"] = -1
    df["confidence"] = -1
    return df


def load_toxtrac(filename_or_buffer, topleft_xy=(0, 0)):
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

    :param filename_or_buffer: Toxtrac results (Tracking_0.txt)
    :param topleft_xy: tuple, length 2; xy coordinates of the arena top left corner
    :return: DataFrame with frame 	id 	x 	y 	width 	height 	confidence columns
    """
    df = pd.read_csv(
        filename_or_buffer,
        delim_whitespace=True,
        names=["frame", "arena", "id", "x", "y", "label"],
        usecols=["frame", "id", "x", "y"],
    )
    df["frame"] += 1  # MATLAB indexing
    df["x"] += topleft_xy[0]
    df["y"] += topleft_xy[1]
    df = df.assign(width=-1)
    df = df.assign(height=-1)
    df = df.assign(confidence=-1)
    df.sort_values(["frame", "id"], inplace=True)
    df[df.isna()] = -1
    return df


def load_sleap_analysis_as_posemot(filename_or_buffer, num_objects=None):
    """

    :param filename_or_buffer:
    :param num_objects:
    :return: PoseMot() nans where object is not present
    """
    import h5py

    f = h5py.File(filename_or_buffer, "r")
    # occupancy_matrix = f['track_occupancy'][:]
    try:
        tracks_matrix = f["tracks"][:]  # noqa: F841
    except KeyError:
        print(
            f'File {filename_or_buffer} doesn\'t appear to be SLEAP "analysis" file.\n'
            f"Export analysis from sleap-label using File -> Export Analysis HDF5.\n"
        )
        raise

    if num_objects is None:
        num_objects = f["tracks"].shape[0]

    mot = PoseMot()
    mot.init_blank(
        range(f["tracks"].shape[3]), range(num_objects), f["tracks"].shape[2]
    )
    mot.ds["x"].values = np.moveaxis(f["tracks"][:num_objects, 0, :, :], 2, 0)
    mot.ds["y"].values = np.moveaxis(f["tracks"][:num_objects, 1, :, :], 2, 0)
    mot.marker_radius = 8
    return mot


def load_sleap_as_dataframe(filename):
    try:
        import sleap
    except ImportError as exception:
        exception.msg = """
io.load_sleap_to_dataframe() requires the sleap module installed. Either install the module or export analysis file from
sleap-label application and use load_posemot_sleap_analysis() without additional dependencies.
        """
        raise exception

    labels = sleap.load_file(filename)

    points = []
    for frame in tqdm.tqdm(labels):
        for instance in frame:
            for node_name, point in zip(labels.skeleton.node_names, instance):
                try:
                    score = point.score
                except AttributeError:
                    score = -1
                if isinstance(instance, sleap.instance.PredictedInstance):
                    instance_class = 'predicted'
                elif isinstance(instance, sleap.instance.Instance):
                    instance_class = 'manual'
                else:
                    assert False, 'unknown instance type: {}'.format(type(instance))
                points.append((point.x, point.y, score, point.visible, node_name, instance.frame_idx,
                               instance.track.name, instance_class, instance.video.backend.filename))

    df = pd.DataFrame(points, columns=['x', 'y', 'score', 'visible', 'bodypart', 'frame',
                                       'track', 'source', 'video'])
    df['keypoint'] = df.bodypart.apply(labels.skeleton.node_names.index)
    return df


def load_sleap_as_posemot(filename):
    df = load_sleap_as_dataframe(filename)
    df['id'] = df.track.str.split('_', expand=True)[1].astype(int)  # default SLEAP track naming "track_<num>"
    df = df.rename(columns={'score': 'confidence'})
    df = df.set_index(["frame", "id", "keypoint"])

    # remove duplicated instance with preference to manually annotated
    df_predicted = df.query('source == "predicted"')
    df_manual = df.query('source == "manual"')
    df_unique = df_predicted.copy()
    df_unique.loc[df_manual.index] = df_manual
    assert df_unique.index.is_unique
    return PoseMot.from_df(df_unique.reset_index())


def save_mot(filename, df):
    df.to_csv(filename, index=False)  # header=False,


def load_mot(filepath_or_buffer):
    """
    Load Multiple Object Tacking Challenge trajectories file.

    :param filepath_or_buffer: mot filename_or_buffer or buffer
    :return: DataFrame, columns frame and id start with 1 (MATLAB indexing)
    """
    df = pd.read_csv(
        filepath_or_buffer, index_col=["frame", "id"]
    )  # names=[u'frame', u'id', u'x', u'y', u'width', u'height', u'confidence']
    return df[(df.x != -1) & (df.y != -1)]


def mot_in_roi(df, roi):
    """
    Limit MOT to a region of interest.

    :param df: MOT trajectories, DataFrame
    :param roi: utils.roi.ROI
    :return: MOT trajectories, DataFrame
    """
    idx_in_roi = (
        (df.x >= roi.x())
        & (df.y >= roi.y())
        & (df.x < roi.x() + roi.width())
        & (df.y < roi.y() + roi.height())
    )
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
    nan_mask = (
        (df_results.x == -1)
        | (df_results.x == -1)
        | df_results.x.isna()
        | df_results.y.isna()
    )
    if len(df_results[nan_mask]) > 0:
        warnings.warn("stripping nans from the evaluated trajectories")
        df_results = df_results[~nan_mask]
    import motmetrics as mm
    from motmetrics.utils import compare_to_groundtruth

    acc = compare_to_groundtruth(
        df_gt, df_results, dist="euc", distfields=["x", "y"], distth=sqdistth
    )
    mh = mm.metrics.create()
    # remove id_global_assignment metric, workaround for https://github.com/cheind/py-motmetrics/issues/19
    metrics = mh.names[:]
    metrics.remove("id_global_assignment")
    return mh.compute(acc, metrics), acc  # metrics=mm.metrics.motchallenge_metrics


def eval_and_save(ground_truth, mot_results, out_csv=None, results_keypoint=None):
    """
    Evaluate results and save metrics.

    :param ground_truth: ground truth filename_or_buffer (MOT format), buffer or Mot object
    :param mot_results: results filename_or_buffer (MOT format), buffer or Mot
    :param out_csv: output file with a summary (filename_or_buffer or buffer)
    :param results_keypoint: keypoint used for evaluation of keypoint/pose data against centroid ground truth
    """
    try:
        df_gt = ground_truth.to_dataframe()
    except AttributeError:
        df_gt = load_mot(ground_truth)
    try:
        df_results = mot_results.to_dataframe()
    except AttributeError:
        df_results = load_any_mot(mot_results).to_dataframe()
    if results_keypoint is not None:
        df_results = df_results[df_results.keypoint == results_keypoint]
    df_gt = df_gt.rename(columns={"frame": "FrameId", "id": "Id"}).set_index(["FrameId", "Id"])
    df_results = df_results.rename(columns={"frame": "FrameId", "id": "Id"}).set_index(["FrameId", "Id"])
    print("Evaluating...")
    summary, acc = eval_mot(df_gt, df_results)
    summary["motp_px"] = np.sqrt(
        summary["motp"]
    )  # convert from square pixels to pixels
    import motmetrics as mm

    # mh = mm.metrics.create()
    print(mm.io.render_summary(summary))
    if out_csv is not None:
        summary.to_csv(out_csv, index=False)


def array_to_mot_dataframe(results):
    """
    Create MOT challenge format DataFrame out of 3 dimensional array of trajectories.

    :param results: ndarray, shape=(n_frames, n_animals, 2 or 4); coordinates are in yx order, nan when id not present
    :return: DataFrame with frame, id, x, y, width, height and confidence columns
    """
    assert results.ndim == 3
    assert results.shape[2] == 2 or results.shape[2] == 4
    objs = []
    columns = ["x", "y"]
    indices = [1, 0]
    if results.shape[2] == 4:
        columns.extend(["width", "height"])
        indices.extend([3, 2])
    for i in range(results.shape[1]):
        df = pd.DataFrame(results[:, i, indices], columns=columns)
        df["frame"] = list(range(1, results.shape[0] + 1))
        df = df[~(df.x.isna() | df.y.isna())]
        df["id"] = i + 1
        df = df[["frame", "id"] + columns]
        objs.append(df)

    df = pd.concat(objs)

    df.sort_values(["frame", "id"], inplace=True)
    if results.shape[2] == 2:
        df["width"] = -1
        df["height"] = -1
    df["confidence"] = -1
    return df
