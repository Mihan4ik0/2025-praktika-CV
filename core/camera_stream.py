import cv2
import numpy as np
import streamlit as st
from core.tracking.sort import Sort

class CameraStream:
    def __init__(self, model):
        self.model = model
        self.tracker = Sort()  # Инициализируем SORT трекер

    def start_stream(self):
        self.tracker = Sort()
        camera = cv2.VideoCapture(0)
        stframe = st.empty()

        if not camera.isOpened():
            st.error("Не удалось открыть камеру.")
            return

        # Создаем кнопку ОДИН РАЗ
        if 'stop_stream' not in st.session_state:
            st.session_state.stop_stream = False

        if st.button("Остановить стрим"):
            st.session_state.stop_stream = True

        while camera.isOpened() and not st.session_state.stop_stream:
            ret, frame = camera.read()
            if not ret:
                st.warning("Не удалось захватить кадр.")
                break

            # Прогоняем кадр через модель
            results = self.model.predict(source=frame, save=False, conf=0.3)

            # Получаем информацию о детекциях
            boxes = results[0].boxes
            cow_boxes = []

            if boxes is not None:
                class_ids = boxes.cls.cpu().numpy().astype(int)

                # Оставляем только коров (class_id == 19)
                cow_mask = (class_ids == 19)
                selected_boxes = boxes.xyxy[cow_mask].cpu().numpy()

                for box in selected_boxes:
                    x1, y1, x2, y2 = box
                    cow_boxes.append([x1, y1, x2, y2, 1.0])  # x1, y1, x2, y2, score

            # Трекинг через SORT
            annotated_frame = frame.copy()

            if cow_boxes:
                tracked_objects = self.tracker.update(np.array(cow_boxes))

                for track in tracked_objects:
                    x1, y1, x2, y2, track_id = track
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    track_id = int(track_id)

                    # Рисуем рамку и подписываем ID
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f'Cow {track_id}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Показываем результат на экране
            stframe.image(annotated_frame, channels="BGR")

        camera.release()
        st.success("Стрим остановлен.")
