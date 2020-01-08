from utils.gt.mot import Mot
import numpy as np
import pandas as pd
import xarray as xr
from shapes.bbox import BBox
import cv2


class BboxMot(Mot):
    def __init__(self, filename=None, **kwds):
        """
        Ground truth stored in xarray.Dataset with frame and id coordinates (frames are 0-indexed).

        Example:

        <xarray.Dataset>
        Dimensions:     (frame: 5928, id: 5)
        Coordinates:
          * frame       (frame) int64 0 1 2 3 4 5 6 ... 5922 5923 5924 5925 5926 5927
          * id          (id) int64 1 2 3 4 5
        Data variables:
            x           (frame, id) float64 434.5 277.7 179.2 180.0 ... nan nan nan nan
            y           (frame, id) float64 279.0 293.6 407.9 430.0 ... nan nan nan nan
            width       (frame, id) float64 nan nan nan nan nan ... nan nan nan nan nan
            height      (frame, id) float64 nan nan nan nan nan ... nan nan nan nan nan
            confidence  (frame, id) float64 1.0 1.0 1.0 1.0 1.0 ... 1.0 1.0 1.0 1.0 1.0
        """
        super(Mot, self).__init__(**kwds)

    def init_blank(self, frames, ids):
        """
        Initialize blank ground truth.

        :param frames: list of frames
        :param ids: list of identities
        """
        self.ds = xr.Dataset(data_vars={'x': (['frame', 'id'], np.nan * np.ones((len(frames), len(ids)))),
                                        'y': (['frame', 'id'], np.nan * np.ones((len(frames), len(ids)))),
                                        'width': (['frame', 'id'], np.nan * np.ones((len(frames), len(ids)))),
                                        'height': (['frame', 'id'], np.nan * np.ones((len(frames), len(ids)))),
                                        'confidence': (['frame', 'id'], np.nan * np.ones((len(frames), len(ids)))),
                                        },
                             coords={'frame': frames, 'id': ids})

    def load(self, filename):
        """
        Load Multiple Object Tacking Challenge trajectories file.

        Format described in https://arxiv.org/abs/1603.00831, section 3.3 Data Format

        Loads trajectories into a DataFrame, columns frame and id start with 1 (MATLAB indexing).

        :param filename: mot filename or buffer
        """
        df = pd.read_csv(filename, index_col=['frame', 'id'],
                         names=['frame', 'id', 'x', 'y', 'width', 'height', 'confidence'],
                         converters={'frame': lambda x: int(x) - 1})
        df[df == -1] = np.nan
        ds = df.to_xarray()
        # ensure that all frames are in the Dataset
        self.init_blank(list(range(ds.frame.min(), ds.frame.max())), ds.id)
        self.ds = ds.merge(self.ds)

    def save(self, filename, make_backup=False):
        import os
        import datetime

        if make_backup and os.path.exists(filename):
            dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            os.rename(filename, filename[:-4] + '_' + dt + '.txt')

        df = self.ds.to_dataframe().reset_index()
        df[df.isna()] = -1
        df['frame'] += 1
        df.to_csv(filename, index=False, header=False)

    def get_bboxes(self, frame):
        """
        Get GT bounding boxes in a frame.

        The returned BBoxes include obj_id attribute.

        :param frame: frame number
        :return: list of bounding boxes (BBox)
        """
        bboxes = []
        for obj_id, obj in self.get_positions_dataframe(frame).iterrows():
            if not (np.isnan(obj.x) or np.isnan(obj.y) or
                    np.isnan(obj.width) or np.isnan(obj.height)):
                bbox = BBox.from_xywh(obj.x, obj.y, obj.width, obj.height, frame)
                bbox.obj_id = obj_id
                bboxes.append(bbox)
        return bboxes

    def get_object_distance(self, frame, obj_id, other):
        """
        TODO bbox iou
        :param frame:
        :param obj_id:
        :param other:
        :return:
        """
        assert False, 'not implemented'

    def draw_frame(self, img, frame, mapping=None):
        """
        Draw objects on an image.

        :param img: ndarray
        :param frame: frame
        :param mapping: mapping of ids, dict
        :return: image
        """
        if frame in self.ds.frame:
            if self.colors is None:
                self._init_draw()
            if mapping is None:
                mapping = dict(list(zip(self.ds.id.data, self.ds.id.data)))
            for bbox in self.get_bboxes(frame):
                bbox.draw_to_image(img, color=self.colors[mapping[bbox.obj_id]])
        return img


