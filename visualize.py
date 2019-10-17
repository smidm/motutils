from moviepy.video.tools.drawing import blit
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip, clips_array
from moviepy.video.VideoClip import ColorClip
from moviepy.video.fx.resize import resize
import numpy as np
import cv2  # TODO: remove dependency


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


def visualize(video_file, out_video_file, trajectories, names=None,
              montage_max_wh=(1920, 1200), duration=None):
    MONTAGE_GRID_WH = [(0, 0), (1, 1), (2, 1), (3, 1), (2, 2), (3, 2),
                       (3, 2)]  # montage grid sizes for 0-6 number of images
    n_clips = len(trajectories)
    clip = VideoFileClip(video_file)
    # n_colors = max([len(t.ds.id) for t in trajectories])  # max number of identities
    # rgb_cycle = generate_colors(n_colors)
    # markers, marker_pos = generate_markers(n_colors, rgb_cycle)

    if names is not None:
        assert len(names) == len(trajectories)
        if 'gt' in names:
            reference_idx = names.index('gt')
        elif 'ground truth' in names:
            reference_idx = names.index('ground truth')
        else:
            reference_idx = 0
    else:
        reference_idx = 0
    mappings = [t.find_mapping(trajectories[reference_idx]) for t in trajectories]

    clips = []
    for i, t in enumerate(trajectories):
        if names is not None:
            name = names[i]
        else:
            name = None
        # text_clip = TextClip(name, size=(200, 100), color='white').set_position('center', 'top')  # , fontsize=100
        video_clip = clip.fl(make_fun(t, name, id_to_gt=mappings[i], fps=clip.fps))
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


    # if args.video_out:
    #     assert args.video_in
    #     visualize_mot(args.video_in, args.video_out, dfs, args.input_names)  # , duration=3)