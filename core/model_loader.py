import os
from ultralytics import YOLO

class ModelLoader:
    def __init__(self, model_name: str = 'yolov8n.pt'):
        os.makedirs("models", exist_ok=True)
        os.environ["ULTRALYTICS_CACHE"] = os.path.join(os.getcwd(), "models")
        self.model = YOLO(model_name)

    def get_model(self):
        return self.model
