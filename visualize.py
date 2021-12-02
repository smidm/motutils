from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip, clips_array
from moviepy.video.VideoClip import ColorClip, TextClip
from moviepy.video.fx.resize import resize
import numpy as np
import cv2  # TODO: remove dependency
from utils.gt.mot import Mot


def process_image(get_frame_fun, t, trajectories, name, id_to_gt, fps):
    """
    Draw single tracker data on a frame.

    :param img:
    :param trajectories:
    :param name:
    :param markers:
    :param marker_pos:
    :param counter:
    :param id_to_gt: list or dict, maps df ids to ground truth ids, used to display same markers for all results
    :return:
    """
    # if id_to_gt is None:
    #     id_to_gt = range(len(markers))  # identity mapping
    img = get_frame_fun(t).copy()
    frame = int(round((t * fps)))
    img = trajectories.draw_frame(img, frame, id_to_gt)
    if name is not None:
        font_size = 1.5
        font_thickness = 2
        text_size, _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)
        cv2.putText(img, name, (int((img.shape[1] - text_size[0]) / 2), 60),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), font_thickness)
    cv2.putText(img, str(frame), (30, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    return img


def process_image_behavior(get_frame_fun, t, mot, trapezoids, fronts, contacts, name, id_to_gt, fps):
    """
    Draw single tracker data on a frame.

    :param img:
    :param trajectories:
    :param name:
    :param markers:
    :param marker_pos:
    :param counter:
    :param id_to_gt: list or dict, maps df ids to ground truth ids, used to display same markers for all results
    :return:
    """
    # if id_to_gt is None:
    #     id_to_gt = range(len(markers))  # identity mapping
    img = get_frame_fun(t).copy()
    frame = int(round((t * fps)))

    for i in range(mot.num_ids()):
        if trapezoids[frame][i] is not None:
            trapezoids[frame][i].draw_to_image(img, color=(255, 0, 0), label=str(i))
        if np.count_nonzero(contacts[frame, i] == 1) > 0:
            fronts[frame][i].draw_to_image(img, color=(255, 255, 0))

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


def make_fun(trajectories, name, id_to_gt=None, fps=None):
    return lambda gf, t: process_image(gf, t, trajectories, name, id_to_gt, fps)


def make_fun_behavior(mot, trapezoids, fronts, contacts, name=None, id_to_gt=None, fps=None):
    return lambda gf, t: process_image(gf, t, mot, trapezoids, fronts, name, id_to_gt, fps)


def process_image_(get_frame, time, fun, fps):
    img = get_frame(time).copy()
    frame = int(round((time * fps)))
    return fun(img, frame)


def plot_frame_to_plot_time(fun, fps):
    return lambda get_frame, time: process_image_(get_frame, time, fun, fps)


def visualize(video_file, out_video_file, funs, names=None,
              montage_max_wh=(1920, 1200), duration=None, montage=True, write_video=True, start_end_frame=None):
    if start_end_frame is None:
        start_end_frame = (0, None)
    MONTAGE_GRID_WH = [(0, 0), (1, 1), (2, 1), (3, 1), (2, 2), (3, 2),
                       (3, 2)]  # montage grid sizes for 0-6 number of images
    if montage:
        n_clips = len(funs)
    else:
        n_clips = 1
    clip = VideoFileClip(video_file)

    if names is not None:
        assert len(names) == len(funs)
        if 'gt' in names:
            reference_idx = names.index('gt')
        elif 'ground truth' in names:
            reference_idx = names.index('ground truth')
        else:
            reference_idx = 0
    else:
        reference_idx = 0
    # mappings = [t.find_mapping(trajectories[reference_idx]) for t in trajectories]

    clips = []
    for i, fun in enumerate(funs):
        if montage:
            if names is not None:
                clips.append(CompositeVideoClip([
                    clip.fl(plot_frame_to_plot_time(fun, clip.fps)),
                    TextClip(names[i], fontsize=100, color='white').set_position(('center', 'top'))
                ]))  # , use_bgclip=True))
            else:
                clips.append(clip.fl(plot_frame_to_plot_time(fun, clip.fps)))
        else:
            clip = clip.fl(plot_frame_to_plot_time(fun, clip.fps))

    if montage:
        n_montage_cells = MONTAGE_GRID_WH[n_clips][0] * MONTAGE_GRID_WH[n_clips][1]
        for _ in range(n_montage_cells - n_clips):
            clips.append(ColorClip(clips[-1].size, (0, 0, 0), duration=clips[-1].duration))
        out_clip = clips_array(np.array(clips).reshape((MONTAGE_GRID_WH[n_clips][1], -1)))
    else:
        out_clip = clip

    newsize = limit_size(out_clip.size, montage_max_wh)
    if newsize is not None:
        out_clip = out_clip.fx(resize, newsize)

    if duration is not None:
        out_clip = out_clip.set_duration(duration)

    start_end = [frame / clip.fps if frame is not None else None for frame in start_end_frame]
    out_clip = out_clip.subclip(*start_end)

    if write_video:
        out_clip.write_videofile(out_video_file)  # , threads=4
    else:
        return out_clip


if __name__ == '__main__':
    import utils.gt.io as io
    import argparse

    parser = argparse.ArgumentParser(description='Visualize mot trajectories.')
    parser.add_argument('video_in', type=str, help='input video file')
    parser.add_argument('video_out', type=str, help='write visualization(s) to a video file')
    parser.add_argument('--load-tox', type=str, help='load ToxTracker trajectories (e.g., Tracking_0.txt)')
    parser.add_argument('--tox-topleft-xy', nargs='+', type=int, help='position of the arena top left corner, see first tuple in the Arena line in Stats_1.txt')
    parser.add_argument('--load-idtracker', type=str, help='load IdTracker trajectories (e.g., trajectories.txt)')
    parser.add_argument('--load-idtrackerai', type=str, help='load idtracker.ai trajectories (e.g., trajectories_wo_gaps.npy)')
    parser.add_argument('--load-mot', type=str, nargs='+', help='load multiple object trajectories file(s)')
    parser.add_argument('--names', type=str, nargs='+', help='names of input files')
    args = parser.parse_args()

    mots = []
    if args.load_tox:
        if not args.tox_topleft_xy:
            parser.error('specify position of the arena top left corner using --tox-topleft-xy')
        if len(args.tox_topleft_xy) != 2:
            parser.error('need to pass exactly two values with --tox-topleft-xy')
        mots.append(Mot.from_df(io.load_toxtrac(args.load_tox, topleft_xy=args.tox_topleft_xy)))

    if args.load_idtracker:
        mots.append(Mot.from_df(io.load_idtracker(args.load_idtracker)))

    if args.load_idtrackerai:
        mots.append(Mot.from_df(io.load_idtrackerai(args.load_idtrackerai)))

    if args.load_mot:
        mots.extend([io.load_any_mot(filename) for filename in args.load_mot])

    if not mots:
        parser.error('no input trajectories specified, see --load options')

    def plot_mot(mot):
        return lambda img, frame: mot.draw_frame(img, frame)

    visualize(args.video_in, args.video_out,
              [plot_mot(mot) for mot in mots],
              args.names) # , duration=1)


