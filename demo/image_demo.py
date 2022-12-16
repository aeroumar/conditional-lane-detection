import torch
import mmcv
from argparse import ArgumentParser
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import cv2

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    cfg = mmcv.Config.fromfile(args.config)
    model = init_detector(cfg, args.checkpoint, device=args.device)

    # cfg = mmcv.Config.fromfile(args.config)
    # # set cudnn_benchmark
    # if cfg.get('cudnn_benchmark', False):
    #     torch.backends.cudnn.benchmark = True
    # cfg.model.pretrained = None    
    # # build the model and load checkpoint
    # model = build_detector(cfg.model)
    # load_checkpoint(model, args.checkpoint, map_location='cpu')
    # # model = MMDataParallel(model, device_ids=[0])


    # test a single image
    loaded_img = cv2.imread(args.img)
    result = inference_detector(model, loaded_img)
    # show the results
    show_result_pyplot(model, loaded_img, result, score_thr=args.score_thr)


if __name__ == '__main__':
    main()
