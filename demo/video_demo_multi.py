# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from typing import Sequence
import sys
from pathlib import Path

import cv2
import numpy as np
from numpy import ndarray
from tqdm import tqdm
import yaml

from mmflow.apis import inference_model, init_model
from mmflow.datasets import visualize_flow

try:
    import imageio
except ImportError:
    imageio = None


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('video', help='video file')
    parser.add_argument('config', type=Path, help='Config yaml file')
    parser.add_argument('out', type=Path, help='output folder')
    parser.add_argument('--filter',  action="store_true", default=False,
                        help="Use only Red channel")
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args


def main(args):
    args.out.mkdir(exist_ok=True, parents=True)

    with open(args.config, 'rt') as file:
        demo_config = yaml.safe_load(file)

    for model in tqdm(demo_config):
        model_name = str(Path(model["path"]).name).split("_")[0]
        # build the model from a config file and a checkpoint file
        model = init_model(model["config"], model["path"], device=args.device)
        # load video
        in_cap = cv2.VideoCapture(args.video)
        assert in_cap.isOpened(), f'Failed to load video file {args.video}'
        
        # get video info
        out_size = (2 * int(in_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(in_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(in_cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(in_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # init video writer
        out_cap = cv2.VideoWriter(str(args.out / f"{model_name}.mp4"), fourcc, fps, out_size, True)

        prev_frame = None
        frame_idx = -1
        with tqdm(total=total_frames) as pbar:
            while (in_cap.isOpened()):
                frame_idx += 1
                pbar.update(1)
                # Get frames
                flag, cur_frame = in_cap.read()
                if not flag:
                    break
                if prev_frame is None:
                    prev_frame = cur_frame
                    continue
                if frame_idx % fps != 0:
                    continue
                
                # estimate flow
                result = inference_model(model, prev_frame, cur_frame)

                if args.filter:
                    dx = result[:, :, 0].copy()
                    dy = result[:, :, 1].copy()
                    angle = np.arctan2(-dy, -dx) / np.pi
                    ignore_inds = ( (angle < 0.45) | (angle > 0.55) )

                    dx[ignore_inds] = 0
                    dy[ignore_inds] = 0

                    result[:, :, 0] = dx.copy()
                    result[:, :, 1] = dy.copy()

                flow_map = visualize_flow(result, None)
                # visualize_flow return flow map with RGB order
                flow_map = cv2.cvtColor(flow_map, cv2.COLOR_RGB2BGR)
                
                result = np.concatenate((cur_frame, flow_map), axis=1)

                out_cap.write(result)
                
                prev_frame = cur_frame

        in_cap.release()
        out_cap.release()


if __name__ == '__main__':
    args = parse_args()
    main(args)
