from __future__ import print_function
import numpy as np
import pandas as pd
import xarray as xr
from shapes.bbox import BBox


class GT(object):
    """
    Ground truth handling.

    When working with a FERDA project, don't forget to set spatial and temporal offsets, see set_project_offsets().

    None means not defined.

    """
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
        self.ds = None

        self.bbox_size_px = None
        self.bbox_match_minimal_iou = 0.5

        if filename is not None:
            self.load(filename)

        super(GT, self).__init__(**kwds)  # this calls potential mixin classes init methods
                                    # see https://stackoverflow.com/a/6099026/322468

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

        :param filename: mot filename
        """
        df = pd.read_csv(filename, index_col=['frame', 'id'],
                         names=[u'frame', u'id', u'x', u'y', u'width', u'height', u'confidence'],
                         converters={u'frame': lambda x: int(x) - 1})
        df[df == -1] = np.nan
        ds = df.to_xarray()
        # ensure that all frames are in the Dataset
        self.init_blank(range(ds.frame.min(), ds.frame.max()), ds.id)
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
        df.to_csv(filename, index=False) # , header=False)

    def print_statistics(self):
        print('counts of number of object ids in frames:')
        print(np.unique(self.ds['x'].count('id').values, return_counts=True))

        print('frames with number of objects other than 0 or num_ids:')
        count_of_valid_ids = self.ds['x'].count('id')
        print(self.ds.sel({'frame': ~count_of_valid_ids.isin([0, self.num_ids()])}))

    def num_ids(self):
        return len(self.ds['id'])

    def get_roi(self):
        """
        Return GT rectangular bounds.

        :return: xmin, xmax, ymin, ymax
        """
        return [float(x) for x in [self.ds['x'].min(), self.ds['x'].max(), self.ds['y'].min(), self.ds['y'].max()]]

    def add_delta(self, delta_x=0, delta_y=0, delta_frames=0):
        """
        Shift ground truth positions and frame numbers by deltas.
        """
        self.ds['x'] += delta_x
        self.ds['y'] += delta_y
        self.ds['frame_'] = self.ds.indexes['frame'] + delta_frames
        self.ds = self.ds.swap_dims({'frame': 'frame_'})
        del self.ds['frame']
        self.ds = self.ds.rename({'frame_': 'frame'})

    def get_positions(self, frame):
        """

        :param frame:
        :return: xarray.Dataset
        """
        return self.ds.sel({'frame': frame})

    def get_xy_numpy(self, frame):
        """

        :param frame:
        :return: ndarray, shape=(n, 2)
        """
        return self.get_positions(frame)[['x', 'y']].to_array().values.T

    def get_positions_dataframe(self, frame):
        """

        :param frame:
        :return: DataFrame, indexed by id, with columns x, y, width, height, confidence
        """
        return self.get_positions(frame).to_dataframe()

    def get_bboxes(self, frame):
        """
        Get GT bounding boxes in a frame.

        The returned BBoxes include obj_id attribute.

        :param frame: frame number
        :return: list of bounding boxes (BBox)
        """
        assert 'bbox_size_px' in dir(self) and self.bbox_size_px is not None
        bboxes = []
        for obj_id, obj in self.get_positions_dataframe(frame).iterrows():
            if not np.isnan(obj.x):
                if np.isnan(obj.width) or np.isnan(obj.height):
                    width = self.bbox_size_px
                    height = self.bbox_size_px
                else:
                    width = obj.width
                    height = obj.height
                bbox = BBox.from_xycenter_hw(obj.x, obj.y, width, height, frame)
                bbox.obj_id = obj_id
                bboxes.append(bbox)
        return bboxes

    def match_bbox(self, query_bbox):
        """
        Match query bounding box to the ground truth.

        The returned BBox includes obj_id attribute.

        :param query_bbox: BBox with defined frame
        :return: None if false positive, best matching BBox otherwise
        """
        bboxes = self.get_bboxes(query_bbox.frame)
        ious = np.array([bbox.iou(query_bbox) for bbox in bboxes])
        if ious.max() < self.bbox_match_minimal_iou:
            return None  # fp
        else:
            return bboxes[ious.argmax()]

    def get_matching_obj_id(self, query_bbox):
        """
        Match query bounding box to the ground truth and return the matching gt object id.

        :param query_bbox:
        :return: object id or None
        """
        matching_bbox = self.match_bbox(query_bbox)
        if matching_bbox is not None:
            return matching_bbox.obj_id
        else:
            return None

    def set_position(self, frame, obj_id, x, y, confidence=1.0):
        self.ds['x'].loc[{'frame': frame, 'id': obj_id}] = x
        self.ds['y'].loc[{'frame': frame, 'id': obj_id}] = y
        self.ds['confidence'].loc[{'frame': frame, 'id': obj_id}] = confidence

    def match_xy(self, frame, xy, maximal_match_distance=None):
        """
        Match query xy to the ground truth.

        :param xy: tuple
        :return: None if false positive, best matching gt row
        """
        distance_vectors = self.ds.sel({'frame': frame}).to_dataframe()[['x', 'y']] - xy
        distances = np.sqrt((distance_vectors['x']**2 + distance_vectors['y']**2))
        matching_id = distances.idxmin()
        if maximal_match_distance is None or distances[matching_id] <= maximal_match_distance:
            return self.ds.sel({'frame': frame, 'id': matching_id})
        else:
            return None  # fp

    def min_frame(self):
        return int(self.ds['frame'].min())

    def max_frame(self):
        return int(self.ds['frame'].max())

    def interpolate_positions(self, frames=None, ids=None):
        """
        Interpolate missing (nan) positions.

        :param frame: list of frame numbers or None for all frames
        :param ids: list of ids or None for all ids
        :return: xarray.Dataset with selected frames and ids and interpolated nans
        """
        if frames is None:
            frames = self.ds.frame
        if ids is None:
            ids = self.ds.id
        # totally inefficient, but really simple: interpolates all nans in selected ids
        return self.ds.sel({'id': ids}).interpolate_na(dim='frame', use_coordinate=True).sel({'frame': frames})

    def _get_index(self):
        return pd.MultiIndex.from_product([range(min(self.df.index.levels[0]), max(self.df.index.levels[0]) + 1),
                                           range(1, self.num_ids + 1)], names=['frame', 'id'])

    def get_missing_positions(self):
        """
        Return frame and id pairs that are not defined in the ground truth.

        :return: DataFrame with frame and id columns
        """
        count_of_valid_ids = self.ds['x'].count('id')
        return self.ds.sel({'frame': ~count_of_valid_ids.isin([5])})

    def draw(self, frames=None, ids=None, marker=None):
        import matplotlib.pylab as plt
        if frames is None:
            frames = self.ds['frame'].values
        if len(frames) == 1 and marker is None:
            marker = 'o'
        if ids is None:
            ids = self.ds['id'].values
        for obj_id in ids:
            pos = self.ds.sel({'frame': frames, 'id': obj_id})
            if not pos['x'].isnull().all() and not pos['y'].isnull().all():
                plt.plot(pos['x'], pos['y'], label=obj_id, marker=marker)


class GtPermutationsMixin(object):
    def __init__(self):
        self.__permutation = {}
        self.__gt_id_to_real_permutation = {}
        for id_ in range(self.__num_ids):
            self.__gt_id_to_real_permutation[id_] = id_
            self.__permutation[id_] = id_

    def set_permutation_reversed(self, data):
        self.__permutation = self.get_permutation(data)
        temp = dict(self.__permutation)
        for key, val in temp.iteritems():
            self.__permutation[val] = key
            self.__gt_id_to_real_permutation[key] = val

    def set_permutation(self, data):
        """
        given list of tuples (frame, id, y, x)
        set internal permutation to fit given experiment

        Args:
            data:

        Returns:

        """

        self.__permutation = self.get_permutation(data)
        for key, val in self.__permutation.iteritems():
            self.__gt_id_to_real_permutation[val] = key

    def get_permutation_reversed(self):
        return self.__gt_id_to_real_permutation

    def get_permutation_dict(self):
        return self.__permutation

    def get_permutation(self, data):
        perm = {}
        for frame, id_, y, x in data:
            original_id_, _ = self.match_gt(frame, y, x)
            perm[id_] = original_id_

        return perm

    def permute(self, data):
        if isinstance(data, list):
            new_data = [None for _ in range(len(data))]
            for i, it in enumerate(data):
                new_data[self.__permutation[i]] = it

            return new_data
        elif isinstance(data, int):
            return self.__permutation[data]
        else:
            return None
