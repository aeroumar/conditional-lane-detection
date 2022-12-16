#!/usr/bin/env python
import os
import cv2
import time
import mmcv
import torch
import rospy
import argparse
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmcv.parallel import MMDataParallel
from tools.condlanenet.common import tusimple_convert_formal
from tools.condlanenet.post_process import CondLanePostProcessor

from mmcv.parallel import collate    
from aerovect_msgs.msg import LaneMetadata
from mmdet.datasets.pipelines import Compose

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

class LaneDetectionNode():
    def __init__(self):
        self.activated = True
        self.input_frame_ =  ImageType()
        # Inference node subscribes to the camera frames topic
        self.camera_frames_sub = rospy.Subscriber("/camera/frame", ImageType, self.inference_clk_)
        # TODO: Find: #############################################################
        # 1- Find topic for getting camera frames in ros
        #   a- For testing: Create command line image publisher
        # 2- Look up what the Image message type is that the frame would come in
        # 3- -----Define the inference callback which runs inference and outputs tusimple lanes 
        # 4. Remove args parser, replace with config and model as params (from launch file?)
        # 5. Move inference code and necessary files into av-common.
        #   a- Build fully then remove uncessary files or Add necessary and then make it work
        # TODO: Setting queue size to eten incase messages pile up, ask is this a valid concern?
        self.lane_detections_pub = rospy.Publisher("/lane_detection/lane_metadata", LaneMetadata, queue_size=10)

        args = self.parse_args()
        # args gets config, checkpoint, local_rank, hm_thr
        self.hm_thr = 0.5
        cfg = mmcv.Config.fromfile(args.config)
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.model.pretrained = None

        # build the model and load checkpoint
        model = build_detector(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location='cpu')
        model = MMDataParallel(model, device_ids=[0])
        model.config = cfg
        model.eval()
        post_processor = CondLanePostProcessor(
            hm_thr=self.hm_thr, mask_size=cfg.mask_size, use_offset=True)        
        self.run(model, post_processor)

    def inference_clk_(self, image_frame_msg):
        self.input_frame_ = image_frame_msg

    def parse_args(self):
        parser = argparse.ArgumentParser(description='MMDet test detector')
        parser.add_argument('config', help='test config file path')
        parser.add_argument('checkpoint', help='seg checkpoint file')
        parser.add_argument('--local_rank', type=int, default=0)
        parser.add_argument('--hm_thr', type=float, default=0.5)
        args = parser.parse_args()
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(args.local_rank)
        return args


    def adjust_result(self, lanes, crop_bbox, img_shape):
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

    def single_gpu_test(self, seg_model, data, original_img, post_processor,
                        hm_thr=0.3,
                        mask_size=(1, 40, 100),
                        crop_bbox=(0, 160, 1280, 720),
                        ):
        
        with torch.no_grad():
            sub_name = data['img_metas'].data[0][0]['sub_img_name']
            img_shape = data['img_metas'].data[0][0]['img_shape']
            ori_shape = data['img_metas'].data[0][0]['ori_shape']
            # h_samples = data['img_metas'].data[0][0]['h_samples']
            h_samples = [240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]
            
            seeds, _ = seg_model(
                return_loss=False, rescale=False, thr=hm_thr, **data)
            downscale = data['img_metas'].data[0][0]['down_scale']
            lanes, seeds = post_processor(seeds, downscale)
            result = self.adjust_result(
                lanes=lanes, crop_bbox=crop_bbox, img_shape=img_shape)
            tusimple_lanes = tusimple_convert_formal(
                result, h_samples, ori_shape[1])
            tusimple_sample = dict(
                lanes=tusimple_lanes,
                h_samples=h_samples,
                raw_file=sub_name,
                run_time=20)  
            return tusimple_sample


    def run(self, model, post_processor):

        # TODO: Feed subscribed message (numpy array of  loaded image instead of loading from file)
        test_path = "/media/aerovect/T7/atl_del_tusimple_lane_dataset/left/8388.jpg"
        loaded_img = cv2.imread(test_path)
        times = []    
        r = rospy.Rate(1)
        while not rospy.is_shutdown():
            if self.input_frame_:
                rospy.loginfo("Running lane detection inference on image frame")
                # TODO: Complete prototype:
                pre_processed_img = self.input_frame_
                loaded_img = pre_processed_img

                test_pipeline = [LoadImage()] + model.config.data.test.pipeline[:]
                test_pipeline = Compose(test_pipeline)
                # prepare data
                img_data = dict(img=loaded_img)
                img_data = test_pipeline(img_data)
                img_data = collate([img_data], samples_per_gpu=1)    
                
                st = time.time()
                tusimple_sample = self.single_gpu_test(model, img_data, loaded_img, post_processor,
                    hm_thr=self.hm_thr,
                    mask_size=model.config.mask_size,
                    crop_bbox=model.config.crop_bbox,
                    )
                times.append(time.time() - st)
                print("\nAverage inference time: ", round(sum(times) / len(times), 5) , 'seconds')     
                # TODO: Convert tusimple sample into LaneMetadata object

                # TODO: Publish LaneMetadata message
            r.sleep()

if __name__ == '__main__':
    rospy.init_node("ADSB Exchange Node", log_level=rospy.DEBUG)
    node = LaneDetectionNode()
    rospy.spin()   
    
    a = LaneDetections()
    a.run()