import tqdm
import pandas as pd
from utils.gt.posegt import PoseGt
from utils.gt.gt_project import GtProject


class PoseGtProject(PoseGt):
    def __init__(self, project=None, **kwargs):
        self.project = project
        super(PoseGtProject, self).__init__(**kwargs)

    @classmethod
    def from_gt_and_regions(cls, project, gt_filename):
        centroid_gt = GtProject(gt_filename)
        centroid_gt.set_project_offsets(project)
        pose_gt = cls(project=project)
        pose_gt.init_blank(centroid_gt.ds.frame, centroid_gt.ds.id, 2)

        missing_regions = []
        non_single_regions = []
        for frame, region_ids in tqdm.tqdm(centroid_gt.match_on_data(project, match_on='regions').items()):  # frames=range(20),
            for obj_id, region_id in zip(pose_gt.ds.id.values, region_ids):
                if region_id is not None:
                    region = project.rm[region_id]
                    if centroid_gt.get_region_cardinality(project, region) == 'single':
                        head_yx, tail_yx = region.get_head_tail()
                        pose_gt.set_position(frame, obj_id, 0, head_yx[1], head_yx[0])
                        pose_gt.set_position(frame, obj_id, 1, tail_yx[1], tail_yx[0])
                    else:
                        non_single_regions.append((frame, obj_id))
                else:
                    missing_regions.append((frame, obj_id))

        print('non single regions')
        print(pd.DataFrame(non_single_regions, columns=['frame', 'id']))
        print('missing regions')
        print(pd.DataFrame(missing_regions, columns=['frame', 'id']))
        return pose_gt


if __name__ == '__main__':
    from core.project.project import Project
    project_paths = [
        # '../projects/2_temp/5Zebrafish_nocover_22min/190828_1819',
        # '../projects/2_temp/Cam1_clip/190828_1819',
        # '../projects/2_temp/Camera3-5min/190828_1819',
        '../projects/2_temp/Sowbug3_cut/190828_1819']
    gt_filenames = [
        # 'data/GT/5Zebrafish_nocover_22min.txt',
        # 'data/GT/Cam1_clip.avi.txt',
        # 'data/GT/Camera3-5min.mp4.txt',
        'data/GT/Sowbug3_cut.txt']
    out_filenames = [
        # 'data/GT/5Zebrafish_nocover_22min_pose.csv',
        # 'data/GT/Cam1_clip.avi_pose.csv',
        # 'data/GT/Camera3-5min.mp4_pose.csv',
        'data/GT/Sowbug3_cut_pose.csv']
            
    for project_path, gt_filename, out_filename in zip(project_paths, gt_filenames, out_filenames):
        print(out_filename)
        p = Project(project_path)
        posegt = PoseGtProject.from_gt_and_regions(p, gt_filename)
        posegt.save(out_filename)
