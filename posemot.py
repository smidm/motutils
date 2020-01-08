import numpy as np
import xarray as xr
import pandas as pd
from shapes.bbox import BBox
from utils.gt.mot import Mot
from itertools import product


class PoseMot(Mot):
    def __init__(self, **kwargs):
        super(PoseMot, self).__init__(**kwargs)

    def init_blank(self, frames, ids, n_points=1):
        """
        Initialize blank ground truth.

        :param frames: list of frames
        :param ids: list of identities
        :param n_points: number of points per object
        """
        assert n_points > 0
        data = {'x': (['frame', 'id', 'keypoint'], np.nan * np.ones((len(frames), len(ids), n_points))),
                'y': (['frame', 'id', 'keypoint'], np.nan * np.ones((len(frames), len(ids), n_points))),
                'confidence': (['frame', 'id', 'keypoint'], np.nan * np.ones((len(frames), len(ids), n_points))),
                }
        self.ds = xr.Dataset(data_vars=data, coords={'frame': frames, 'id': ids, 'keypoint': list(range(n_points))})

    # def load(self, filename):
    #     pass

    # def get_xy_numpy(self, frame):
    #     """
    #
    #     :param frame:
    #     :return: ndarray, shape=(n, 2)
    #     """
    #     return self.get_positions(frame)[['x', 'y']].to_array().values.T

    def load(self, filename):
        df = pd.read_csv(filename, index_col=['frame', 'id', 'keypoint'],
                         converters={'frame': lambda x: int(x) - 1})
        df[df == -1] = np.nan
        ds = df.to_xarray()
        # ensure that all frames are in the Dataset
        self.init_blank(range(ds.frame.min().item(), ds.frame.max().item()), ds.id, len(ds.keypoint))
        self.ds = ds.merge(self.ds)

    def get_obj_roi(self, frame, obj_id):
        """
        Get object extreme points.

        :param frame: int
        :param obj_id: int
        :return: xmin, xmax, ymin, ymax
        """
        obj = self.ds.sel(dict(frame=frame, id=obj_id))
        return [float(val) for val in [obj['x'].min(), obj['x'].max(), obj['y'].min(), obj['y'].max()]]

    # def isnan(self, frame, obj_id):
    #     return bool(np.isnan(self.ds.sel(dict(frame=frmae, id=obj_id))['x']).all())

    def get_bboxes(self, frame):
        """
        Get GT bounding boxes in a frame.

        The returned BBoxes include obj_id attribute.

        :param frame: frame number
        :return: list of bounding boxes (BBox)
        """
        bboxes = []
        for obj_id in self.ds.id.values:
            xmin, xmax, ymin, ymax = self.get_obj_roi(frame, obj_id)
            if not any(np.isnan([xmin, xmax, ymin, ymax])):
                bbox = BBox(xmin, ymin, xmax, ymax, frame)
                bbox.obj_id = obj_id
                bboxes.append(bbox)
        return bboxes

    def set_position(self, frame, obj_id, keypoint, x, y, confidence=1.0):
        """
        Set single keypoint position.

        :param frame:
        :param obj_id:
        :param x:
        :param y:
        :param confidence: object position confidence
        :param kwargs: 'x1', 'y1', 'x2', 'y2', ... xy positions for further points
        """
        self.ds['confidence'].loc[{'frame': frame, 'id': obj_id, 'keypoint': keypoint}] = confidence
        self.ds['x'].loc[{'frame': frame, 'id': obj_id, 'keypoint': keypoint}] = x
        self.ds['y'].loc[{'frame': frame, 'id': obj_id, 'keypoint': keypoint}] = y

    # def set_object_position(self, frame, obj_id, xys, confidence=1.0):

    def match_xy(self, frame, xy, max_match_distance_px=None):
        """
        Match query keypoints to the ground truth.

        :param frame:
        :param xy: keypoint coords, array, shape=(n_keypoints, 2)
        :param maximal_match_distance: TODO
        :return: None if false positive, best matching gt row
        """
        # <xarray.DataArray ([x, y], id, keypoint)>
        xy_diff_sq = ((self.ds.sel(dict(frame=frame))[['x', 'y']] - xy.T) ** 2).to_array()

        matching_id = xy_diff_sq.sum(['keypoint', 'variable']).argmin(dim='id')
        if max_match_distance_px is None or \
            all(np.sqrt(xy_diff_sq.sel(dict(id=matching_id)).sum('variable')) <= max_match_distance_px):
            return self.ds.sel({'frame': frame, 'id': matching_id})
        else:
            return None

    def draw_frame(self, img, frame, mapping=None):
        from moviepy.video.tools.drawing import blit
        if frame in self.ds.frame:
            if self.markers is None or self.marker_position is None:
                self._init_draw(marker_radius=3)
            if mapping is None:
                mapping = dict(list(zip(self.ds.id.data, self.ds.id.data)))
            for obj_id, keypoint_id in product(self.ds.id.data, self.ds.keypoint.data):
                row = self.ds.sel(dict(frame=frame, id=obj_id, keypoint=keypoint_id))
                if not (np.isnan(row.x) or np.isnan(row.y)):
                    marker = self.markers[mapping[obj_id]]
                    img = blit(marker['img'], img, (int(row.x) - self.marker_position[0],
                                                    int(row.y) - self.marker_position[1]),
                               mask=marker['mask'])
        return img

    def get_object_distance(self, frame, obj_id, other):
        self_pos = self.get_object(frame, obj_id)
        return np.sqrt(((self_pos[['x', 'y']] - other[['x', 'y']]).to_array() ** 2).sum())

    # def draw(self, frames=None, ids=None, marker=None):
    #     import matplotlib.pylab as plt
    #     if frames is None:
    #         frames = self.ds['frame'].values
    #     if len(frames) == 1 and marker is None:
    #         marker = 'o'
    #     if ids is None:
    #         ids = self.ds['id'].values
    #     for obj_id in ids:
    #         pos = self.ds.sel({'frame': frames, 'id': obj_id})
    #         if not all(pos['x'].isnull()) and not all(pos['y'].isnull()):
    #             plt.plot(pos['x'], pos['y'], label=obj_id, marker=marker)

