from typing import Any, Callable, List, Optional, Tuple, Union

try:
    from numpy.typing import ArrayLike
except ModuleNotFoundError:
    ArrayLike = Any
import os

import cv2  # TODO: remove dependency
import numpy as np
from moviepy.video.compositing.CompositeVideoClip import (CompositeVideoClip,
                                                          clips_array)
from moviepy.video.fx.resize import resize
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import ColorClip, TextClip


def process_image(get_frame_fun, t, trajectories, name, id_to_gt, fps):
    """
    Draw single tracker data on a frame.
    """
    # if id_to_gt is None:
    #     id_to_gt = range(len(markers))  # identity mapping
    img = get_frame_fun(t).copy()
    frame = int(round((t * fps)))
    img = trajectories.draw_frame(img, frame, id_to_gt)
    if name is not None:
        font_size = 1.5
        font_thickness = 2
        text_size, _ = cv2.getTextSize(
            name, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness
        )
        cv2.putText(
            img,
            name,
            (int((img.shape[1] - text_size[0]) / 2), 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255, 255, 255),
            font_thickness,
        )
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
        newsize = 1.0 / size_ratio.max()
        assert np.all(wh * newsize <= max_wh)
    return newsize


def visualize(
    video_file: Union[str, bytes, os.PathLike],
    out_video_file: Optional[Union[str, bytes, os.PathLike]],
    funs: List[Callable[[ArrayLike, int], ArrayLike]],
    names: Optional[List[str]] = None,
    montage_max_wh: Tuple[int, int] = (1920, 1200),
    duration: Optional[float] = None,
    montage: bool = True,
    start_end_frame: Optional[Tuple[Optional[int], Optional[int]]] = None,
) -> VideoFileClip:
    """
    Render a video with annotated objects.

    Supports:
    - multiple annotation sources
    - montage or overlaid visualization types for comparison of multiple annotations

    :param video_file: input video filename
    :param out_video_file: optional output video filename
    :param funs: list of drawing functions with signature of Mot.draw_frame()
    :param names: optional list of names of annotation sources
    :param montage_max_wh: maximum video size in pixels
    :param duration: maximal duration in seconds
    :param montage: if multiple annotation sources in funs and True: render montage of videos,
                    if multiple annotation sources in funs and False render overlaid annotations over a single video
    :param start_end_frame: optional specification of start and end frame of visualization
    :return: visualization video clip object
    """
    def plot_frame_to_plot_time(fun: Callable[[ArrayLike, int], ArrayLike],
                                fps: float) -> Callable[[Callable, float], ArrayLike]:
        """
        Change callable signature from fun(img, frame) -> img
                                    to fun(get_frame_fun, t) -> img
        :param fun: fun(img, frame) -> img, signature of Mot.draw_frame()
        :param fps: clip frames per second
        :return: signature of moviepy.VideoClip.fl: fun(get_frame_fun, t) -> img
        """
        def clip_fl_fun(get_frame_fun, t):
            img = get_frame_fun(t).copy()
            frame = int(round((t * fps)))
            return fun(img, frame)

        return clip_fl_fun

    if start_end_frame is None:
        start_end_frame = (0, None)
    MONTAGE_GRID_WH = [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 1),
        (2, 2),
        (3, 2),
        (3, 2),
    ]  # montage grid sizes for 0-6 number of images
    if montage:
        n_clips = len(funs)
    else:
        n_clips = 1
    clip = VideoFileClip(video_file)
    clips = []
    for i, fun in enumerate(funs):
        if montage:
            if names is not None:
                clips.append(
                    CompositeVideoClip(
                        [
                            clip.fl(plot_frame_to_plot_time(fun, clip.fps)),
                            TextClip(
                                names[i], fontsize=100, color="white"
                            ).set_position(("center", "top")),
                        ]
                    )
                )  # , use_bgclip=True))
            else:
                clips.append(clip.fl(plot_frame_to_plot_time(fun, clip.fps)))
        else:
            clip = clip.fl(plot_frame_to_plot_time(fun, clip.fps))

    if montage:
        n_montage_cells = MONTAGE_GRID_WH[n_clips][0] * MONTAGE_GRID_WH[n_clips][1]
        for _ in range(n_montage_cells - n_clips):
            clips.append(
                ColorClip(clips[-1].size, (0, 0, 0), duration=clips[-1].duration)
            )
        out_clip = clips_array(
            np.array(clips).reshape((MONTAGE_GRID_WH[n_clips][1], -1))
        )
    else:
        out_clip = clip

    newsize = limit_size(out_clip.size, montage_max_wh)
    if newsize is not None:
        out_clip = out_clip.fx(resize, newsize)

    if duration is not None:
        out_clip = out_clip.set_duration(duration)

    start_end = [
        frame / clip.fps if frame is not None else None for frame in start_end_frame
    ]
    out_clip = out_clip.subclip(*start_end)

    if out_video_file:
        out_clip.write_videofile(out_video_file)  # , threads=4

    return out_clip
