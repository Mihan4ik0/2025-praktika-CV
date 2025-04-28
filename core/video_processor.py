import torch
import cv2
import numpy as np
from core.tracking.sort import Sort


class VideoProcessor:
    def __init__(self, model):
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)  # Оставляем перенос модели на нужное устройство (GPU/CPU)

    def preprocess_frame(self, frame):
        """Приведение кадра к размеру кратному 32."""
        height, width, _ = frame.shape
        new_height = (height // 32) * 32
        new_width = (width // 32) * 32
        return cv2.resize(frame, (new_width, new_height))

    def predict_frame(self, frame_resized):
        """Прогон кадра через модель."""
        results = self.model.predict(source=frame_resized, save=False, conf=0.3)
        return results

    def _process_frame_for_tracking(self, frame):
        """
        Обработка одного кадра для трекинга:
        - ресайз
        - предсказание
        - выбор только коров
        - подготовка боксов
        """
        frame_resized = self.preprocess_frame(frame)
        results = self.predict_frame(frame_resized)

        boxes_detected = []

        boxes = results[0].boxes
        if boxes is not None:
            class_ids = boxes.cls.cpu().numpy().astype(int)
            cow_mask = (class_ids == 19)

            if cow_mask.any():
                selected_boxes = boxes.xyxy[cow_mask].cpu().numpy()
                for box in selected_boxes:
                    x1, y1, x2, y2 = box
                    boxes_detected.append([x1, y1, x2, y2, 1.0])  # bbox + confidence 1.0

        return boxes_detected, frame_resized

    def process_video(self, video_path: str):
        """Простая обработка видео без трекинга (для быстрой отрисовки всех детекций)."""
        cap = cv2.VideoCapture(video_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = self.preprocess_frame(frame)
            results = self.predict_frame(frame_resized)
            annotated_frame = results[0].plot()
            frames.append(annotated_frame)

        cap.release()
        return frames

    def detect_objects_in_video(self, video_path: str) -> dict:
        """Подсчёт уникальных коров в видео с помощью трекинга."""
        self.tracker = Sort()
        cap = cv2.VideoCapture(video_path)

        unique_cow_ids = set()
        total_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1

            boxes_detected, _ = self._process_frame_for_tracking(frame)

            if boxes_detected:
                tracked_objects = self.tracker.update(np.array(boxes_detected))

                for track in tracked_objects:
                    track_id = int(track[4])
                    unique_cow_ids.add(track_id)

        cap.release()

        total_cows = len(unique_cow_ids)
        average_cows_per_frame = total_cows / total_frames if total_frames else 0

        return {
            "total_cows": int(total_cows),
            "average_cows_per_frame": round(float(average_cows_per_frame), 2)
        }

    def stream_video_with_detection(self, video_path: str):
        """Стриминг обработки видео с правильным подсчётом уникальных коров."""
        self.tracker = Sort()
        self.unique_cow_ids = set()

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frames += 1
            boxes_detected, frame_resized = self._process_frame_for_tracking(frame)

            frame_annotated = frame_resized.copy()

            if boxes_detected:
                tracked_objects = self.tracker.update(np.array(boxes_detected))

                for track in tracked_objects:
                    x1, y1, x2, y2, track_id = track
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    track_id = int(track_id)

                    # Отрисовка
                    cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_annotated, f'Cow {track_id}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Добавляем ID коровы
                    self.unique_cow_ids.add(track_id)

            # ВОТ ЗДЕСЬ: считаем общее количество уникальных коров
            total_cows = len(self.unique_cow_ids)

            yield frame_annotated, total_cows, processed_frames, total_frames

        cap.release()


