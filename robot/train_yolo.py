"""シミュレーターで作った画像で、積み木と箱を見つける YOLO を学習する。
  python robot/train_yolo.py   →  robot/runs/blocks/weights/best.pt
"""
import os

from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    m = YOLO(os.path.join(HERE, "..", "models", "yolo", "yolo26n.pt"))
    m.train(data=os.path.join(HERE, "data", "blocks.yaml"), epochs=40, imgsz=640, batch=16, workers=2,
            project=os.path.join(HERE, "runs"), name="blocks", exist_ok=True, plots=False,
            fliplr=0.5, flipud=0.5, degrees=10, mosaic=1.0, close_mosaic=5, verbose=False)
