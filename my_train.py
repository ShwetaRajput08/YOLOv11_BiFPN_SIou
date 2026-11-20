import argparse
import os
import torch
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Optimize GPU computation performance
torch.backends.cudnn.benchmark = True


def parse_opt():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--epochs", type=int, default=1, help="number of epochs"
  )
  parser.add_argument("--batch", type=int, default=32, help="batch size")
  parser.add_argument("--workers", type=int, default=4, help="number of workers")
  parser.add_argument(
      "--device",
      type=str,
      default="0",
      help="cuda device, i.e. 0, 0,1,2,3 or cpu",
  )
  parser.add_argument("--resume", action="store_true", help="resume training")
  opt = parser.parse_args()
  return opt


if __name__ == "__main__":
  opt = parse_opt()

  # Robust device assignment for GPU / CUDA environments
  if torch.cuda.is_available() and opt.device.lower() != "cpu":
    device = int(opt.device) if opt.device.isdigit() else opt.device
    print(f"========================================")
    print(f"🚀 Training on GPU: {torch.cuda.get_device_name(0)}")
    print(f"🔥 PyTorch CUDA Runtime Version: {torch.version.cuda}")
    print(f"========================================")
  else:
    device = "cpu"
    print("⚠️ CUDA is unavailable or 'cpu' was specified. Training on CPU.")
#First run on yolo11_BiFPN.yaml for 200 epoches saved in run13 on Flood dataset

  # Load custom model configuration and weights
  model = YOLO(model=r"..\ultralytics\cfg\models\11\yolo11s-hsfpn-adaslicer.yaml")
  model.load("yolo11s.pt")

  # Start training (amp=False bypasses the pre-check asset download error)
  model.train(
      data=r"..\ultralytics\cfg\FloodDatasets\flood.yaml",
      imgsz=640,
      epochs=opt.epochs,
      patience=20,  # Stop early if no improvement
      batch=opt.batch,
      workers=opt.workers,
      device=device,
      optimizer="SGD",
      close_mosaic=10,
      resume=opt.resume,
      project="runs/train",
      name="exp",
      single_cls=False,
      cache=False,
      amp=False,  # <--- Added to bypass the asset check error
  )