from pathlib import Path

import click
import imageio
import numpy as np
import tqdm

from . import io
from . import visualize as motutils_visualize
from .mot import Mot


@click.group()
@click.option(
    "--load-mot",
    multiple=True,
    type=click.File(),
    help="load a MOT challenge csv file(s)",
)
@click.option(
    "--load-gt",
    type=click.File(),
    help="load ground truth from a MOT challenge csv file",
)
@click.option(
    "--load-idtracker",
    multiple=True,
    type=click.File(),
    help="load IdTracker trajectories (e.g., trajectories.txt)",
)
@click.option(
    "--load-idtrackerai",
    multiple=True,
    type=click.File(mode="rb"),
    help="load idtracker.ai trajectories (e.g., trajectories_wo_gaps.npy)",
)
@click.option(
    "--load-sleap-analysis",
    multiple=True,
    type=click.File(mode="rb"),
    help="load SLEAP analysis trajectories (exported from sleap-label File -> Export Analysis HDF5)",
)
@click.option(
    "--load-sleap",
    multiple=True,
    type=click.Path(exists=True, readable=True, dir_okay=False),
    help="load SLEAP trajectories",
)
@click.option(
    "--load-toxtrac",
    multiple=True,
    type=click.File(),
    help="load ToxTracker trajectories (e.g., Tracking_0.txt)",
)
@click.option(
    "--toxtrac-topleft-xy",
    multiple=True,
    type=(int, int),
    help="position of the arena top left corner, see first tuple in the Arena line in Stats_1.txt",
)
@click.pass_context
def cli(
    ctx,
    load_mot,
    load_gt,
    load_idtracker,
    load_idtrackerai,
    load_sleap_analysis,
    load_sleap,
    load_toxtrac,
    toxtrac_topleft_xy,
):
    mots = []
    mots.extend([io.load_any_mot(filename) for filename in load_mot])
    mots.extend(
        [io.load_sleap_analysis_as_posemot(filename) for filename in load_sleap_analysis]
    )
    mots.extend(
        [io.load_sleap_as_posemot(filename) for filename in load_sleap]
    )
    dfs = []
    dfs.extend([io.load_idtracker(filename) for filename in load_idtracker])
    dfs.extend([io.load_idtrackerai(filename) for filename in load_idtrackerai])
    if toxtrac_topleft_xy and len(toxtrac_topleft_xy) != len(load_toxtrac):
        raise click.BadParameter(
            "--toxtrac-topleft-xy has to be specified for all --load-toxtrac input files or omitted entirely"
        )
    elif len(toxtrac_topleft_xy) == 0:
        toxtrac_topleft_xy = [None] * len(load_toxtrac)
    dfs.extend(
        [
            io.load_toxtrac(filename, topleft_xy)
            for filename, topleft_xy in zip(load_toxtrac, toxtrac_topleft_xy)
        ]
    )
    mots.extend([Mot.from_df(df) for df in dfs])
    if load_gt:
        gt = io.load_any_mot(load_gt)
    else:
        gt = None
    ctx.ensure_object(dict)
    ctx.obj["mots"] = mots
    ctx.obj["gt"] = gt


@cli.command()
@click.pass_context
@click.argument(
    "video-in",
    type=click.Path(exists=True, readable=True, dir_okay=False),
)
@click.argument(
    "video-out",
    type=click.Path(exists=False, dir_okay=False),
)
@click.argument("source-display-name", nargs=-1)
@click.option("--limit-duration", help="visualization duration limit in s", default=-1)
@click.option("--montage/--no-montage",
              help="choose between multiple video montage or a single overlaid video",
              default=True)
@click.option('--color-ids', 'visualization_colors', flag_value='ids', default=True)
@click.option('--color-keypoints', 'visualization_colors', flag_value='keypoints')
def visualize(ctx, video_in, video_out, source_display_name, limit_duration, montage, visualization_colors):
    """
    Visualize MOT file(s) overlaid on a video.
    """
    if len(source_display_name) == 0:
        source_display_name = None
    if ctx.obj["gt"]:
        mots = [ctx.obj["gt"]] + ctx.obj["mots"]
    else:
        mots = ctx.obj["mots"]
    if source_display_name and ctx.obj["gt"]:
        names = ["gt"] + source_display_name
    else:
        names = source_display_name
    if names and len(mots) != len(names):
        raise click.BadParameter(
            "source-display-name must be the same length as number of input files"
        )
    if not mots:
        raise click.BadParameter("missing at least one input file")
    visualization_colors_used = False
    if visualization_colors == 'keypoints':
        for mot in mots:
            try:
                mot.visualization_color_by_keypoints = True
                visualization_colors_used = True
            except AttributeError:
                pass
        if not visualization_colors_used:
            raise click.BadParameter(
                "color-keypoints is available only for input data with keypoints"
            )
    # overlaid visualization: create copies of the first mot colormap with decreasing intensity for other mots
    # if not montage and len(mots) > 1:
    #     from matplotlib.colors import ListedColormap
    #     alpha_values = np.linspace(1., 0., len(mots) + 1)[:-1]
    #     for mot, alpha in zip(mots[1:], alpha_values[1:]):
    #         new_cmap = mots[0].colormap(np.arange(mots[0].colormap.N))
    #         new_cmap[:, 0:3] *= alpha
    #         mot.colormap = ListedColormap(new_cmap)

    if not montage and len(mots) == 2:
        draw_funs = [
            mots[0].draw_frame,
            lambda img, frame: mots[1].draw_frame(img, frame, alternative_marker=True),
        ]
    else:
        draw_funs = [mot.draw_frame for mot in mots]

    motutils_visualize(
        video_in,
        video_out,
        draw_funs,
        names,
        montage=montage,
        duration=limit_duration if limit_duration != -1 else None,
    )


@cli.command()
@click.pass_context
@click.argument(
    "output-mot",
    type=click.File(mode="w"),
)
def convert(ctx, output_mot):
    """
    Convert any format to MOT Challenge format.
    """
    if len(ctx.obj["mots"]) != 1:
        raise click.BadParameter("convert requires exactly single input file")
    ctx.obj["mots"][0].save(output_mot)


@cli.command()
@click.pass_context
@click.option(
    "--write-eval",
    type=click.File(mode="w"),
    help="write evaluation results as a CSV file",
)
@click.option(
    "--keypoint",
    type=int,
    default=-1,
    help="keypoint to use when evaluating pose MOT results against point ground truth",
)
def eval(ctx, write_eval, keypoint):
    """
    Evaluate a single MOT file against the ground truth.
    """
    if len(ctx.obj["mots"]) != 1:
        raise click.BadParameter("eval requires exactly single input file")
    if not ctx.obj["gt"]:
        raise click.BadParameter("eval requires ground truth (--load-gt)")
    io.eval_and_save(
        ctx.obj["gt"],
        ctx.obj["mots"][0],
        write_eval,
        keypoint if keypoint != -1 else None,
    )
