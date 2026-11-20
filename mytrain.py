import argparse
import torch
from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of workers')
    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--resume', action='store_true', help='resume training')
    # 可以添加其他需要的参数
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()

    # Automatically fall back to CPU only if CUDA is unavailable
    device = opt.device if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")

    model = YOLO(model=r'..\ultralytics\cfg\models\11\yolo11_BiFPN.yaml')
    model.load('yolo11s.pt')
    #C:\Users\user\PycharmProjects\YOLOv11_BiFPN_SIou\ultralytics\cfg\FloodDatasets\flood.yaml
    model.train(data=r'..\ultralytics\cfg\FloodDatasets\flood.yaml',
                imgsz=640,
                epochs=opt.epochs,
                patience=20, # new line added to stop early if no improvment SNR
                batch=opt.batch,
                workers=opt.workers,
                device=device,
                optimizer='SGD',
                close_mosaic=10,
                resume=opt.resume,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                )