from shapes.bbox import BBox
import numpy as np


class GtDummyDetectorMixin(object):
    def __init__(self, bb_size_px=None, fp_prob=0, fn_prob=0,
                 true_detection_jitter_scale=None,
                 false_detection_jitter_scale=None, **kwargs):
        self.detector_bb_size_px = bb_size_px
        self.detector_fp_prob = fp_prob  # probability of false positive detection per actual object in gt
        self.detector_fn_prob = fn_prob
        if true_detection_jitter_scale is None:
            self.true_scale = bb_size_px / 8
        else:
            self.true_scale = true_detection_jitter_scale
        if false_detection_jitter_scale is None:
            self.false_scale = bb_size_px
        else:
            self.false_scale = false_detection_jitter_scale
        super(GtDummyDetectorMixin, self).__init__(**kwargs)

    def detect(self, frame_nr):
        bboxes = []
        bb_half_px = self.detector_bb_size_px / 2
        gt_detections = self.get_positions(frame_nr).to_dataframe()
        for obj_id, det in gt_detections.iterrows():
            if np.random.rand() > self.detector_fn_prob:
                xy = [det.x, det.y] + np.random.normal(scale=self.true_scale, size=(2,))
                bbox = BBox(*np.concatenate((xy - bb_half_px, xy + bb_half_px)), frame=frame_nr)
                bbox.obj_id = obj_id
                bboxes.append(bbox)
            if np.random.rand() < self.detector_fp_prob:
                xy = [det.x, det.y] + np.random.normal(scale=self.false_scale, size=(2,))
                bbox = BBox(*np.concatenate((xy - bb_half_px, xy + bb_half_px)), frame=frame_nr)
                bbox.obj_id = None
                bboxes.append(bbox)
        return bboxes


