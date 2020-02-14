import numpy as np
import pandas as pd
import xarray as xr
from shapes.bbox import BBox
import scipy.optimize
import warnings
from collections import Counter
import numbers
import tqdm


class Mot(object):
    """
    Multiple object trajectories persistence, visualization, interpolation, analysis.

    Single object is represented by its centroid.
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
            confidence  (frame, id) float64 1.0 1.0 1.0 1.0 1.0 ... 1.0 1.0 1.0 1.0 1.0
        """
        self.ds = None

        self.bbox_size_px = None
        self.bbox_match_minimal_iou = 0.5

        # see draw_frame()
        self.marker_position = None
        self.markers = None

        if filename is not None:
            self.load(filename)

        super(Mot, self).__init__(**kwds)  # this calls potential mixin classes init methods
                                    # see https://stackoverflow.com/a/6099026/322468

    def init_blank(self, frames, ids):
        """
        Initialize blank ground truth.

        :param frames: list of frames
        :param ids: list of identities
        """
        self.ds = xr.Dataset(data_vars={'x': (['frame', 'id'], np.nan * np.ones((len(frames), len(ids)))),
                                        'y': (['frame', 'id'], np.nan * np.ones((len(frames), len(ids)))),
                                        'confidence': (['frame', 'id'], np.nan * np.ones((len(frames), len(ids)))),
                                        },
                             coords={'frame': frames, 'id': ids})

    def load(self, filename):
        """
        Load trajectories of multiple objects from CSV file.

        -1 are replaced by nans

        Columns:
        - frame (first frame is 1, is converted to 0 during loading)
        - id (arbitrary numbered)
        - x
        - y
        - confidence

        :param filename: mot filename or buffer
        """
        df = pd.read_csv(filename, index_col=['frame', 'id'],
                         converters={'frame': lambda x: int(x) - 1})
        df[df == -1] = np.nan
        ds = df.to_xarray()
        # ensure that all frames are in the Dataset
        self.init_blank(range(ds.frame.min().item(), ds.frame.max().item()), ds.id)
        self.ds = ds.merge(self.ds)

    def save(self, filename, float_precision=1):
        df = self.ds.to_dataframe().reset_index()
        df[df.isna()] = -1
        df['frame'] += 1
        df.to_csv(filename, index=False, float_format='%.' + str(float_precision) + 'f')

    def save_via_json(self, filename, video_filename, fps, description=None):
        from . import via
        import json
        json_out = via.via_json(video_filename, description)
        json_out['attribute'] = {1: via.attribute('locations', 'FILE1_Z2_XY0', 'TEXT'),
                                 2: via.attribute('missing', 'FILE1_Z2_XY0', 'TEXT'),
                                 3: via.attribute('id', 'FILE1_Z1_XY1', 'RADIO',
                                                  options=range(self.num_ids()))}
        for frame in tqdm.tqdm(range(self.min_frame(), self.max_frame() + 1)):
            for obj_id, row in self.get_positions_dataframe(frame).iterrows():
                if np.isnan(row['x']) or np.isnan(row['y']):
                    continue
                json_out['metadata'].update((via.metadata(round(frame * 1/fps, 5), 'POINT',
                                                          [round(x, 2) for x in row[['x', 'y']]],
                                                          {'2': str(obj_id)}),))
        # for frame in self.get_missing_positions().frame.values:
        #     json_out['metadata'].update((via.metadata((round(frame * 1/fps, 5), round((frame + 2) * 1/fps, 5))),))

        pos = self.get_missing_positions()
        for frame in pos.frame.values:
            pos_frame = pos.sel({'frame': frame})
            for i in pos_frame.id.values:
                json_out['metadata'].update(
                    (via.metadata((round(frame * 1 / fps, 5), round((frame + 2) * 1 / fps, 5)),
                                  attributes={'2': str(i)}),))

        with open(filename, 'w') as fw:
            json.dump(json_out, fw)


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

    def get_object(self, frame, obj_id):
        return self.ds.sel(dict(frame=frame, id=obj_id))

    def get_xy_numpy(self, frame):
        """

        :param frame:
        :return: ndarray, shape=(n, 2)
        """
        return self.get_positions(frame)[['x', 'y']].to_array().values.T

    def get_positions_dataframe(self, frame):
        """

        :param frame:
        :return: DataFrame, indexed by id, with columns x, y, confidence
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
            if not (np.isnan(obj.x) or np.isnan(obj.y)):
                bbox = BBox.from_xycenter_wh(obj.x, obj.y, self.bbox_size_px, self.bbox_size_px, frame)
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
        return pd.MultiIndex.from_product([list(range(min(self.df.index.levels[0]), max(self.df.index.levels[0]) + 1)),
                                           list(range(1, self.num_ids + 1))], names=['frame', 'id'])

    def get_missing_positions(self):
        """
        Return frame and id pairs that are not defined in the ground truth.

        :return: DataFrame with frame and id columns
        """
        count_of_valid_ids = self.ds['x'].count('id')
        return self.ds.sel({'frame': ~count_of_valid_ids.isin([self.num_ids()])})

    def speed(self):
        """
        Return object speed on frame basis.

        :return: distance travelled between frames; xarray.Dataset, coordinates: frame, id
        """
        return np.sqrt(self.ds.x.diff('frame') ** 2 + self.ds.y.diff('frame') ** 2)

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

    def _init_draw(self, marker_radius=10):
        import matplotlib.pylab as plt
        from moviepy.video.tools.drawing import circle
        cm = plt.get_cmap('gist_rainbow')
        self.colors = dict(list(zip(self.ds.id.data,
                               [cm(1. * i / len(self.ds.id), bytes=True)[:3] for i in range(len(self.ds.id))])))

        blur = marker_radius * 0.2
        img_dim = marker_radius * 2 + 1
        img_size = (img_dim, img_dim)
        self.marker_position = (marker_radius, marker_radius)
        self.markers = {}
        for obj_id, c in list(self.colors.items()):
            img = circle(img_size, self.marker_position, marker_radius, c, blur=blur)
            mask = circle(img_size, self.marker_position, marker_radius, 1, blur=blur)
            self.markers[obj_id] = {'img': img, 'mask': mask}

    def draw_frame(self, img, frame, mapping=None):
        """
        Draw objects on an image.

        :param img: ndarray
        :param frame: frame
        :param mapping: mapping of ids, dict
        :return: image
        """
        from moviepy.video.tools.drawing import blit
        if frame in self.ds.frame:
            if self.markers is None or self.marker_position is None:
                self._init_draw()
            if mapping is None:
                mapping = dict(list(zip(self.ds.id.data, self.ds.id.data)))
            for obj_id in self.ds.id.data:
                row = self.ds.sel(dict(frame=frame, id=obj_id))
                if not (np.isnan(row.x) or np.isnan(row.y)):
                    marker = self.markers[mapping[obj_id]]
                    img = blit(marker['img'], img, (int(row.x) - self.marker_position[0],
                                                    int(row.y) - self.marker_position[1]),
                               mask=marker['mask'])
        return img

    def get_object_distance(self, frame, obj_id, other):
        self_pos = self.ds.sel(dict(frame=frame, id=obj_id))
        return np.linalg.norm((self_pos[['x', 'y']] - other[['x', 'y']]).to_array())

    def find_mapping(self, other, n_frames_to_probe=10):
        """
        Find spatially close mapping to other set of trajectories.

        Probe n_frames_to_probe frames, most frequent mapping is returned.

        :param other: other trajectories
        :param n_frames_to_probe: number of random frames to match
        :return: dict, self ids to other ids
        """
        assert len(self.ds.id) == len(other.ds.id)
        # get positions in a suitable frame
        self_all_ids = self.ds.where(self.ds.x.count(dim='id') == len(self.ds.id), drop=True)
        other_all_ids = other.ds.where(other.ds.x.count(dim='id') == len(other.ds.id), drop=True)
        frames = np.intersect1d(self_all_ids.frame, other_all_ids.frame, assume_unique=True)
        jjs = Counter()
        for frame in np.random.choice(frames, n_frames_to_probe):
            # match positions
            distance_matrix = np.vectorize(lambda i, j: self.get_object_distance(frame, i, other.get_object(frame, j))) \
                (*np.meshgrid(self.ds.id, other.ds.id, indexing='ij'))
            ii, jj = scipy.optimize.linear_sum_assignment(distance_matrix)
            if np.count_nonzero(distance_matrix[ii, jj] > 10):
                warnings.warn('large distance beween detection and gt ' + str(distance_matrix[ii, jj]))
            jjs[tuple(jj)] += 1

        jj = np.array(jjs.most_common()[0][0])
        self_to_other = dict(list(zip(self.ds.id[ii].data, other.ds.id[jj].data)))
        # add identity mappings for ids not present in the selected frame
        # all_ids = df.index.get_level_values(1).unique()
        # if len(ids) < len(all_ids):
        #     ids_without_gt_match = set(all_ids) - set(ids)
        #     self_to_other.update(zip(ids_without_gt_match, ids_without_gt_match))  # add identity mapping
        return self_to_other


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
        for key, val in list(temp.items()):
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
        for key, val in list(self.__permutation.items()):
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
        elif isinstance(data, numbers.Integral):
            return self.__permutation[data]
        else:
            return None
