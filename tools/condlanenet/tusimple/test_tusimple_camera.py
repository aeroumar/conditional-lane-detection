import argparse
import os
import numpy as np
import math
import json
import cv2
import copy
import mmcv
import torch
import torch.distributed as dist
import PIL.Image
import PIL.ImageDraw
from mmcv import Config
from mmdet.apis import init_detector, inference_detector
from mmcv.parallel import MMDataParallel
from mmcv.runner import init_dist, load_checkpoint
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.utils.general_utils import mkdir, path_join
from tools.condlanenet.common import tusimple_convert_formal, COLORS
from tools.condlanenet.post_process import CondLanePostProcessor
from torch.utils.data import IterableDataset, Dataset, DataLoader
import matplotlib.pyplot as plt
import pyzed.sl as sl

# class MyIterableDataset(IterableDataset):
    # def __init__(self, img):
    #     dataset_type = 'TuSimpleDataset'
    #     self.testDict = dict(
    #                     self.type=dataset_type,
    #                     self.data_list=[img],
    #                     self.pipeline=val_pipeline,
    #                     self.test_mode=True)
    # def __iter__(self):
    #     return self.testDict

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='seg checkpoint file')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--hm_thr', type=float, default=0.5)
    parser.add_argument('--show', action='store_true')
    parser.add_argument(
        '--show_dst',
        default='/media/harish/T71/data_collection/atl_del_processed/output',
        help='path to save visualized results.')
    parser.add_argument(
        '--result_dst',
        default='./work_dirs/tusimple/results',
        help='path to save results')
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


def out_result(lanes, dst=None):
    if dst is not None:
        with open(dst, 'w') as f:
            for lane in lanes:
                for idx, p in enumerate(lane):
                    if idx == len(lane) - 1:
                        print('{:.2f} '.format(p[0]), end='', file=f)
                        print('{:.2f}'.format(p[1]), file=f)
                    else:
                        print('{:.2f} '.format(p[0]), end='', file=f)
                        print('{:.2f} '.format(p[1]), end='', file=f)


def vis_one(img, results, filename, img_info=None, lane_width=7):

    # def parse_img_info(img_info):
    #     converted_lanes = []
    #     h_samples = img_info['h_samples']
    #     lanes = img_info['lanes']
    #     for lane in lanes:
    #         converted_lane = []
    #         for coord_x, coord_y in zip(lane, h_samples):
    #             if coord_x >= 0:
    #                 converted_lane.append((coord_x, coord_y))
    #         converted_lanes.append(converted_lane)
    #     return converted_lanes

    # img = cv2.imread(filename)
    #img_gt = cv2.imread(filename)
    img_pil = PIL.Image.fromarray(img)
    #img_gt_pil = PIL.Image.fromarray(img_gt)
    for idx, lane in enumerate(results):
        lane_tuple = [tuple(p) for p in lane]
        PIL.ImageDraw.Draw(img_pil).line(
            xy=lane_tuple, fill=COLORS[idx + 1], width=lane_width)
    img_new = np.array(img_pil, dtype=np.uint8)
    '''
    if img_info is not None:
        gt_lanes = parse_img_info(img_info)
        for idx, lane in enumerate(gt_lanes):
            lane_tuple = [tuple(p) for p in lane]
            PIL.ImageDraw.Draw(img_gt_pil).line(
                xy=lane_tuple, fill=COLORS[idx + 1], width=lane_width)
        img_gt = np.array(img_gt_pil, dtype=np.uint8)
    '''
    cv2.imwrite(filename, img_new)
    return img_new, #img_gt


def single_gpu_test(seg_model,
                    #zed_cam,
                    show=None,
                    hm_thr=0.3,
                    result_dst=None,
                    nms_thr=4,
                    mask_size=(1, 80, 200),
                    crop_bbox=(0, 160, 1280, 720),
                    ):

    #cap = cv2.VideoCapture(4)
    image = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    # if not cap.isOpened():
    #     print("Cannot open camera")
    #     exit()
    #seg_model.eval()
    #dataset = data_loader.dataset
    post_processor = CondLanePostProcessor(
        hm_thr=hm_thr, mask_size=mask_size, use_offset=True)
    #prog_bar = mmcv.ProgressBar(len(dataset))
    if result_dst is not None:
        mkdir(result_dst)
        dst_dir = os.path.join(result_dst, 'test.json')
        f_dst = open(dst_dir, 'w')
    count = 0
    cap = cv2.VideoCapture(4)
    if cap.isOpened() == 0:
        exit(-1)

    # Set the video resolution to HD720 (2560*720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while True:
        # Capture frame-by-frame
        #ret, frame = cap.read()
        #plt.title('matplotlib.pyplot.imshow() function Example', 
                                   #  fontweight ="bold")

        # if zed_cam.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # # A new image is available if grab() returns SUCCESS
        #     zed_cam.retrieve_image(image, sl.VIEW.LEFT)
        #     image_ocv = image.get_data()
        #     image_ocv = image_ocv[:,:,0:3]
        retval, frame = cap.read()
        
        left_right_image = np.split(frame, 2, axis=1)
        image_ocv = left_right_image[1]
        
        if not retval:
            print("Can't receive frame (stream end?). Exiting ...")
            break
            
        dest = '/home/harish/Documents/conditional-lane-detection/output/' + str(count) + '.jpg'
        # cv2.imwrite(dest, image_ocv)
        count += 1

        #dataset = MyIterableDataset(image_ocv)
        # data_loader = build_dataloader(
        #             dataset,
        #             samples_per_gpu=1,
        #             workers_per_gpu=4,
        #             dist=True,
        #             shuffle=False)
        # if frame is read correctly ret is True
        # if not ret:
        #     print("Can't receive frame (stream end?). Exiting ...")
        #     break
        h_samples = [240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]
        img_shape = (320, 800, 3)
        ori_shape = (720, 1280, 3)
        downscale = 4
        #img_info = data['img_metas'].data[0][0]['img_info']
        #print(img_info)
        #for i, data in enumerate(data_loader):
        with torch.no_grad():
            #sub_name = data['img_metas'].data[0][0]['sub_img_name']
            # img_shape = data['img_metas'].data[0][0]['img_shape']
            # ori_shape = data['img_metas'].data[0][0]['ori_shape']
            # h_samples = data['img_metas'].data[0][0]['h_samples']
            #img_info = data['img_metas'].data[0][0]['img_info']
            seeds, _ = inference_detector(seg_model, image_ocv)
            #downscale = data['img_metas'].data[0][0]['down_scale']
            lanes, seeds = post_processor(seeds, downscale)
            result = adjust_result(
                lanes=lanes, crop_bbox=crop_bbox, img_shape=img_shape)
            if result_dst is not None:
                mkdir(result_dst)
                dst_dir = os.path.join(result_dst, 'test.json')
                tusimple_lanes = tusimple_convert_formal(
                    result, h_samples, ori_shape[1])
                tusimple_sample = dict(
                    lanes=tusimple_lanes,
                    h_samples=h_samples,
                    # raw_file=sub_name,
                    run_time=20)
                #json.dump(tusimple_sample, f_dst)
                print(file=f_dst)

            # filename = data['img_metas'].data[0][0]['filename']
        # img_pil = PIL.Image.fromarray(image_ocv)
        # #img_gt_pil = PIL.Image.fromarray(img_gt)
        # for idx, lane in enumerate(result):
        #     lane_tuple = [tuple(p) for p in lane]
        #     PIL.ImageDraw.Draw(img_pil).line(
        #         xy=lane_tuple, fill=COLORS[idx + 1], width=7)
        # img = np.array(img_pil, dtype=np.uint8)
        # print('_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_IMAGE_VIS SHAPE_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+')
        # print(img.type)
        #cv2.imshow('lane lines', img)
        
        if show is not None and show:
            filename = dest
            img_vis = vis_one(image_ocv, result, filename)
            # save_name = sub_name.replace('/', '.')
            # dst_show_dir = path_join(show, save_name)
            # dst_show_gt_dir = path_join(show, save_name + '.gt.jpg')
            
            # cv2.imwrite(filename, img_vis)
            # cv2.imwrite(dst_show_gt_dir, img_gt_vis)
        

        batch_size = 1
        # for _ in range(batch_size):
        #     prog_bar.update()
    
    if result_dst:
        f_dst.close()
    # zed.close()
    cap.release()
    cv2.destroyAllWindows()


class DateEnconding(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, np.float32):
            return float(o)


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    #dataset = build_dataset(cfg.data.test)
    # data_loader = build_dataloader(
    #     dataset,
    #     samples_per_gpu=1,
    #     workers_per_gpu=cfg.data.workers_per_gpu,
    #     dist=distributed,
    #     shuffle=False)

    # build the model and load checkpoint
    model = init_detector('configs/condlanenet/tusimple/tusimple_large_test.py', '/home/harish/Documents/conditional-lane-detection/epoch_35.pth', device='cuda:0')
    #load_checkpoint(model, args.checkpoint, map_location='cpu')
    #model = MMDataParallel(model, device_ids=[0])
    if not args.show:
        show_dst = None
    else:
        show_dst = args.show_dst
    #if args.show is not None and args.show:
        #mkdir(args.show_dst)
    #zed = sl.Camera()
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD1080 video mode
    init_params.camera_fps = 30  # Set fps at 30

    # Open the camera
    # err = zed.open(init_params)
    # if err != sl.ERROR_CODE.SUCCESS:
    #     exit(1)
    single_gpu_test(
        seg_model=model,
        #zed_cam = zed,
        show=show_dst,
        hm_thr=args.hm_thr,
        result_dst=args.result_dst,
        nms_thr=cfg.nms_thr,
        mask_size=cfg.mask_size,
        crop_bbox=cfg.crop_bbox
        )


if __name__ == '__main__':
    main()
