import pandas as pd
import fire


def load_pkl(filename):
    """
    Ferda legacy ground truth storage.

    :param filename: ground truth pkl
    :return: mot DataFrame
    """
    from utils.gt.gt import GT
    gt = GT()
    gt.load(filename)
    gt.set_offset()  # this sometimes fixes gt.get_positions()
    data = []
    for frame in range(gt.min_frame(), gt.max_frame()):
        try:
            for i, yx in enumerate(gt.get_positions(frame), 1):
                if yx is None:
                    x = -1
                    y = -1
                else:
                    x = yx[1]
                    y = yx[0]
                data.append([frame + 1, i, x, y, -1, -1, 1])
        except:
            print(frame)
    return pd.DataFrame(data, columns=['frame', 'id', 'x', 'y', 'width', 'height', 'confidence'])


def pkltomot(filename_pkl, filename_mot):
    df = load_pkl(filename_pkl)
    df.to_csv(filename_mot, header=False, index=False)


if __name__ == '__main__':
    fire.Fire(pkltomot)