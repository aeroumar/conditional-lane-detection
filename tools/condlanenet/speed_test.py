import argparse
import os
import cv2
import numpy as np
import mmcv
import torch

from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.utils.general_utils import Timer
from tools.condlanenet.common import tusimple_convert_formal
from mmdet.models.detectors.condlanenet import CondLanePostProcessor

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

from mmcv.parallel import collate
from mmdet.datasets.pipelines import Compose

SIZE = (800, 320)

class LoadImage(object):
    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'checkpoint', default=None, help='test config file path')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def adjust_result(lanes, crop_bbox, img_shape):
    h_img, w_img = img_shape[:2]
    ratio_x = (crop_bbox[2] - crop_bbox[0]) / w_img
    ratio_y = (crop_bbox[3] - crop_bbox[1]) / h_img
    offset_x, offset_y = crop_bbox[:2]

    results = []
    if lanes is not None:
        for key in range(len(lanes)):
            pts = []
            for pt in lanes[key]['points']:
                pt[0] = float(pt[0] * ratio_x + offset_x)
                pt[1] = float(pt[1] * ratio_y + offset_y)
                pts.append(tuple(pt))
            if len(pts) > 1:
                results.append(pts)
    return results

def get_lanes(model, post_processor, mean, std, img):
    
    img = img[270:, ...]
    img = cv2.resize(img, SIZE)
    img = mmcv.imnormalize(img, mean, std, False)
    x = torch.unsqueeze(torch.from_numpy(img).permute(2, 0, 1), 0)
    x = x.cuda()

    pipeline = [{'type': 'albumentation', 'pipelines': [{'type': 'Compose', 'params': {'bboxes': False, 'keypoints': True, 'masks': False}}, {'type': 'Crop', 'x_min': 0, 'x_max': 1280, 'y_min': 160, 'y_max': 720, 'p': 1}, {'type': 'Resize', 'height': 320, 'width': 800, 'p': 1}]}, {'type': 'Normalize', 'mean': [75.3, 76.6, 77.6], 'std': [50.5, 53.8, 54.3], 'to_rgb': False}, {'type': 'ImageToTensor', 'keys': ['img']}, {'type': 'CollectLane', 'down_scale': 4, 'hm_down_scale': 16, 'radius': 6, 'keys': ['img', 'gt_hm'], 'meta_keys': ['filename', 'sub_img_name', 'gt_masks', 'mask_shape', 'hm_shape', 'ori_shape', 'img_shape', 'down_scale', 'hm_down_scale', 'img_norm_cfg', 'gt_points', 'h_samples', 'img_info']}]

    test_pipeline = [LoadImage()] + pipeline
    print("test_pipeline:", test_pipeline)
    test_pipeline = Compose(test_pipeline)
    # prepare data
    img_data = dict(img=img)
    img_data = test_pipeline(img_data)
    data = collate([img_data], samples_per_gpu=1)  

    with torch.no_grad():
        h_samples = [240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]
        
        seeds, _ = model(
            return_loss=False, rescale=False, thr=0.5, **data)

        lanes, seeds = post_processor(seeds, 4)
        result = adjust_result(
            lanes=lanes, crop_bbox=(0, 160, 1280, 720), img_shape=(320, 800, 3))
        tusimple_lanes = tusimple_convert_formal(
            result, h_samples, (720, 1280, 3)[1])
        tusimple_sample = dict(
            lanes=tusimple_lanes,
            h_samples=h_samples,
            run_time=20)
            
        return tusimple_sample

def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    model = build_detector(cfg.model)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    post_processor = CondLanePostProcessor(mask_size=(1, 80, 200), hm_thr=0.5, seg_thr=0.5)
    model = model.cuda().eval()
    mean = np.array([75.3, 76.6, 77.6])
    std = np.array([50.5, 53.8, 54.3])

    
    base_dir = "/media/aerovect/T7/atl_del_tusimple_lane_dataset/left_small"
    for path in os.listdir(base_dir):
        img = cv2.imread(os.path.join(base_dir, path))
        print("img", img.shape)
        tusimple_sample = get_lanes(model, post_processor, mean, std, img) 
        print("\nImage: ", path, "\nTusimple: \n", tusimple_sample)




if __name__ == '__main__':
    main()