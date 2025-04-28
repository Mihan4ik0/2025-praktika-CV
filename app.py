import streamlit as st
import cv2
import numpy as np
from datetime import datetime, date
import pandas as pd
from collections import defaultdict
import pathlib
import io
import os
import time

# Импорт своих сервисов
from services.file_service import save_unique_file
from services.report_service import generate_pdf
from core.model_loader import ModelLoader
from core.image_processor import ImageProcessor
from core.video_processor import VideoProcessor
from core.camera_stream import CameraStream
from services.history_service import HistoryService

# Проверим рабочие пути и логи
import traceback

# Инициализация сервиса
history_service = HistoryService()

# Функция для вывода лога в Streamlit
def log_error(message):
    st.error(message)
    print(message)


# Настройки страницы
st.set_page_config(page_title="Мониторинг коров", layout="wide")

# Инициализация
model_option = st.sidebar.selectbox("Выберите модель:", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"])
model_loader = ModelLoader(model_option)
model = model_loader.get_model()

image_processor = ImageProcessor(model)
video_processor = VideoProcessor(model)
camera_stream = CameraStream(model)
history_service = HistoryService()

mode = st.sidebar.selectbox("Выберите действие:", ["Обработка изображения", "Обработка видео", "Стрим с камеры", "История запросов"])

# ---------------- Обработка изображения ----------------
if mode == "Обработка изображения":
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        with st.spinner("Обрабатываем изображение..."):
            file_bytes = uploaded_file.read()
            original_save_path = save_unique_file("uploads", uploaded_file.name, file_bytes)

            file_size = os.path.getsize(original_save_path) / 1024  # КБ
            img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            resolution = f"{img.shape[1]}x{img.shape[0]}"

            start_time = time.time()
            annotated_image = image_processor.process_image(file_bytes)
            detection_stats = image_processor.detect_objects_in_image(file_bytes)
            processing_time = round(time.time() - start_time, 2)

            result_image_path = os.path.join("outputs", f"processed_{uploaded_file.name}")
            os.makedirs("outputs", exist_ok=True)
            cv2.imwrite(result_image_path, annotated_image)

            st.image(annotated_image, channels="BGR", use_container_width=True)

            history_service.save_record({
                "type": "image",
                "original_filename": original_save_path,
                "processed_filename": result_image_path,
                "detections": {"cows": int(detection_stats["cows"])} ,
                "file_size_kb": round(file_size, 2),
                "resolution": resolution,
                "processing_time_sec": processing_time,
                "timestamp": datetime.now().isoformat(),
                "model_used": model_option
            })

            with open(result_image_path, "rb") as file:
                st.download_button("📥 Скачать обработанное изображение", data=file, file_name=os.path.basename(result_image_path), mime="image/jpeg")

# ---------------- Обработка видео ----------------
elif mode == "Обработка видео":
    uploaded_video = st.file_uploader("Загрузите видео", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video:
        with st.spinner("Обрабатываем видео..."):
            file_bytes = uploaded_video.read()
            original_save_path = save_unique_file("uploads", uploaded_video.name, file_bytes)

            file_size = os.path.getsize(original_save_path) / 1024  # КБ
            frames = []
            stframe = st.empty()
            progress_bar = st.progress(0)
            cow_counter = st.empty()

            start_time = time.time()
            for frame_annotated, total_cows, processed_frames, total_frames in video_processor.stream_video_with_detection(original_save_path):
                stframe.image(frame_annotated, channels="BGR", use_container_width=True)
                cow_counter.info(f"🐄 Найдено коров всего: {total_cows}")
                progress_bar.progress(min(int((processed_frames / total_frames) * 100), 100))
                frames.append(frame_annotated)

            processing_time = round(time.time() - start_time, 2)
            stframe.empty()
            progress_bar.empty()

            result_video_path = os.path.join("outputs", f"processed_{uploaded_video.name}")
            os.makedirs("outputs", exist_ok=True)

            # Используем XVID кодек для видео
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            height, width, _ = frames[0].shape
            out = cv2.VideoWriter(result_video_path, fourcc, 20.0, (width, height))
            for frame in frames:
                out.write(frame)
            out.release()

            resolution = f"{width}x{height}"

            with open(result_video_path, "rb") as f:
                video_bytes = f.read()
            st.video(io.BytesIO(video_bytes))

            history_service.save_record({
                "type": "video",
                "original_filename": original_save_path,
                "processed_filename": result_video_path,
                "detections": {
                    "total_cows": int(total_cows),
                    "average_cows_per_frame": round(total_cows / processed_frames, 2) if processed_frames else 0
                },
                "file_size_kb": round(file_size, 2),
                "resolution": resolution,
                "processing_time_sec": processing_time,
                "timestamp": datetime.now().isoformat(),
                "model_used": model_option
            })

            with open(result_video_path, "rb") as file:
                st.download_button("📥 Скачать обработанное видео", data=file, file_name=os.path.basename(result_video_path), mime="video/mp4")

# ---------------- Стрим с камеры ----------------
elif mode == "Стрим с камеры":
    camera_stream.start_stream()

# ---------------- История запросов ----------------
elif mode == "История запросов":
    st.header("📜 История обработки файлов")

    history = history_service.load_history()
    filtered_history = []
    total_cows_images = 0
    total_cows_videos = 0
    average_cows_videos = []

    if history:
        st.subheader("🔎 Фильтрация истории")
        file_type_filter = st.selectbox("Тип файлов:", ["Все", "Только изображения", "Только видео"])
        use_date_filter = st.checkbox("Фильтровать по диапазону дат")

        if use_date_filter:
            date_range = st.date_input("Выберите диапазон дат:", [])
        else:
            date_range = None

        if st.button("🔄 Сбросить фильтры"):
            st.rerun()

        for idx, record in enumerate(history):
            record_date = datetime.fromisoformat(record["timestamp"]).date()

            if file_type_filter == "Только изображения" and record["type"] != "image":
                continue
            if file_type_filter == "Только видео" and record["type"] != "video":
                continue

            if use_date_filter and date_range:
                if len(date_range) != 2:
                    st.warning("Пожалуйста, выберите начальную и конечную дату!")
                    continue
                start_date, end_date = date_range
                if not (start_date <= record_date <= end_date):
                    continue

            filtered_history.append((idx, record))

            detections = record.get('detections', {})
            if record["type"] == "image":
                total_cows_images += detections.get("cows", 0)
            else:
                total_cows_videos += detections.get("total_cows", 0)
                average_cows_videos.append(detections.get("average_cows_per_frame", 0))

        if not filtered_history:
            st.info("Нет записей, соответствующих выбранным фильтрам.")
        else:
            if 'show_statistics' not in st.session_state:
                st.session_state.show_statistics = False

            col_stat, col_excel, col_pdf = st.columns(3)
            with col_stat:
                if st.button("📊 Показать/Скрыть общую статистику"):
                    st.session_state.show_statistics = not st.session_state.show_statistics

            with col_excel:
                df = pd.DataFrame([{
                    "Тип": rec["type"].capitalize(),
                    "Оригинал": os.path.basename(rec["original_filename"]),
                    "Результат": os.path.basename(rec["processed_filename"]),
                    "Детекции": rec.get("detections", {}),
                    "Размер (КБ)": round(rec.get("file_size_kb", 0), 2),
                    "Разрешение": rec.get("resolution", "-"),
                    "Время обработки (сек)": rec.get("processing_time_sec", "-"),
                    "Дата/Время": rec.get("timestamp", "-"),
                    "Модель использованная": rec.get("model_used", "-")
                } for idx, rec in filtered_history])

                buffer_excel = io.BytesIO()
                with pd.ExcelWriter(buffer_excel, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name="Отчёт")
                buffer_excel.seek(0)

                st.download_button("📥 Скачать Excel-отчёт", data=buffer_excel, file_name="filtered_history_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            with col_pdf:
                pdf_buffer = generate_pdf([rec for _, rec in filtered_history])
                st.download_button("📄 Скачать PDF-отчёт", data=pdf_buffer, file_name="filtered_history_report.pdf", mime="application/pdf")

            st.divider()

            st.success(f"🔍 Найдено записей: {len(filtered_history)}")

            if st.session_state.show_statistics:
                with st.expander("📈 Общая статистика", expanded=True):
                    st.markdown(f"**🖼️ Всего изображений:** {sum(1 for _, rec in filtered_history if rec['type'] == 'image')}")
                    st.markdown(f"**🎞️ Всего видео:** {sum(1 for _, rec in filtered_history if rec['type'] == 'video')}")
                    st.markdown(f"**🐄 Коров на изображениях:** {total_cows_images}")
                    st.markdown(f"**🐄 Коров на видео:** {total_cows_videos}")
                    if average_cows_videos:
                        avg_per_video = round(sum(average_cows_videos) / len(average_cows_videos), 2)
                        st.markdown(f"**📈 Среднее число коров на кадр в видео:** {avg_per_video}")
                    else:
                        st.markdown("**📈 Среднее число коров на кадр в видео:** 0")

            grouped_history = defaultdict(list)
            for idx, record in filtered_history:
                record_date = datetime.fromisoformat(record["timestamp"]).date()
                grouped_history[str(record_date)].append((idx, record))

            for date_str, records in sorted(grouped_history.items(), reverse=True):
                st.subheader(f"📅 Дата: {date_str}")

                for idx, record in records:
                    with st.container(border=True):
                        col1, col2, col3 = st.columns([2, 2, 1])

                        with col1:
                            icon = "📸" if record["type"] == "image" else "🎥"
                            st.markdown(f"**{icon} Оригинал:** {os.path.basename(record['original_filename'])}")
                            st.markdown(f"**Размер:** {round(record.get('file_size_kb', 0), 2)} КБ")
                            st.markdown(f"**Разрешение:** {record.get('resolution', '-')}")

                        with col2:
                            detections = record.get('detections', {})
                            det_text = (f'Всего: {detections.get("total_cows", 0)} | Ср: {detections.get("average_cows_per_frame", 0)}') if record["type"] == "video" else f'{detections.get("cows", 0)} коров(ы)'
                            st.markdown(f"**🧩 Детекции:** {det_text}")
                            st.markdown(f"**⏱️ Время обработки:** {record.get('processing_time_sec', '-')} сек")
                            st.markdown(f"**🧳 Модель:** {record.get('model_used', '-')}")

                        with col3:
                            if st.button(f"👀 Предпросмотр", key=f"preview_{idx}"):
                                with st.expander(f"Предпросмотр файла: {os.path.basename(record['processed_filename'])}", expanded=True):
                                    col_prev1, col_prev2 = st.columns(2)
                                    with col_prev1:
                                        st.markdown("**🖼️ Оригинал:**")
                                        if record["type"] == "image":
                                            st.image(record['original_filename'], use_container_width=True)
                                        else:
                                            with open(record['original_filename'], "rb") as f:
                                                st.video(io.BytesIO(f.read()))  # Показываем оригинальное видео

                                    with col_prev2:
                                        st.markdown("**🖼️ Результат:**")
                                        if record["type"] == "image":
                                            st.image(record['processed_filename'], use_container_width=True)
                                        else:
                                            with open(record['processed_filename'], "rb") as f:
                                                st.video(io.BytesIO(f.read()))  # Показываем обработанное видео

    if st.button("🧹 Очистить историю и удалить все файлы"):
        history_service.clear_history_and_files()
        st.success("История и файлы удалены. Перезагрузите страницу.")
