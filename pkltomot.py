import pandas as pd
import fire
from utils.gt.gt import GT
import warnings


def load_pkl(filename):
    """
    Ferda legacy ground truth storage.

    :param filename: ground truth pkl
    :return: mot DataFrame
    """
    gt = GT()
    gt.load(filename)
    gt.set_offset(x=0, y=0, frames=0)
    data = []
    for frame in range(gt.min_frame(), gt.max_frame()):
        for i, y_x_type in enumerate(gt.get_positions_and_types(frame), 1):
            if y_x_type is None:
                x = -1
                y = -1
                gt_type = -1
            else:
                y, x, gt_type = y_x_type
            data.append([frame + 1, i, x, y, -1, -1, 1, gt_type])
    return pd.DataFrame(data, columns=['frame', 'id', 'x', 'y', 'width', 'height', 'confidence', 'type'])


def pkltomot(filename_pkl, filename_mot):
    df = load_pkl(filename_pkl)
    df = df[df.type != 0]  # gt type == 0 records are removed
    del df['type']
    df.to_csv(filename_mot, header=False, index=False)


if __name__ == '__main__':
    fire.Fire(pkltomot)
