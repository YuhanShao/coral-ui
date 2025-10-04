# inference.py
import torch
from PIL import Image

class CoralPipeline:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # TODO: 之后把你们的 YOLO / CoralScope 模型加载到这里
        # 例如：
        # from ultralytics import YOLO
        # self.yolo = YOLO("weights/yolo_coral.pt")

    @torch.inference_mode()
    def run(self, img: Image.Image):
        # TODO: 之后在这里做真实的检测+分割，并生成叠加图
        w, h = img.size
        overlay = img.copy()
        results = {
            "detections": [
                {"label": "coral_bleached", "conf": 0.87, "bbox": [int(0.1*w), int(0.2*h), int(0.6*w), int(0.8*h)]}
            ],
            "segmentation": {"exists": True, "coverage_pct": 42.3},
            "meta": {"device": self.device}
        }
        return overlay, results

def load_pipeline():
    return CoralPipeline()
