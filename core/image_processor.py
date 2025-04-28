import torch
import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, model):
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Проверяем доступность GPU
        self.model.to(self.device)  # Перемещаем модель на GPU или CPU

    def process_image(self, file_bytes: bytes) -> np.ndarray:
        # Декодируем изображение
        image = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype=np.uint8), 1)

        # Изменяем размер изображения
        height, width, _ = image.shape
        new_height = (height // 32) * 32
        new_width = (width // 32) * 32
        image_resized = cv2.resize(image, (new_width, new_height))

        # Переводим изображение в тензор и переносим на GPU
        image_tensor = torch.from_numpy(image_resized).to(self.device).float()
        image_tensor /= 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)

        # Прогон через модель
        results = self.model.predict(source=image_tensor, save=False, conf=0.3)

        # Делаем копию изображения для аннотаций
        annotated_image = image_resized.copy()

        boxes = results[0].boxes
        if boxes is not None:
            class_ids = boxes.cls.cpu().numpy().astype(int)
            cow_mask = (class_ids == 19)

            if cow_mask.any():
                selected_boxes = boxes.xyxy[cow_mask].cpu().numpy()
                for box in selected_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_image, "Cow", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return annotated_image


    def detect_objects_in_image(self, file_bytes: bytes) -> dict:
        # Декодируем изображение
        image = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype=np.uint8), 1)

        # Изменяем размер изображения, чтобы оно было кратно 32 (например, 640x640)
        height, width, _ = image.shape
        new_height = (height // 32) * 32
        new_width = (width // 32) * 32
        image_resized = cv2.resize(image, (new_width, new_height))

        # Переводим изображение в тензор и переносим на GPU
        image_tensor = torch.from_numpy(image_resized).to(self.device).float()  # Изменяем тип на float
        image_tensor /= 255.0  # Нормализуем изображение в диапазоне [0, 1]

        # Добавляем batch dimension (BCHW)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # (C, H, W) -> (1, C, H, W)

        # Прогоняем через модель
        results = self.model.predict(source=image_tensor, save=False, conf=0.3)

        boxes = results[0].boxes
        class_ids = boxes.cls.cpu().numpy().astype(int)  # Переносим обратно на CPU

        # Ищем только коров (класс 19)
        cows_count = (class_ids == 19).sum()

        return {"cows": int(cows_count)}
