from fpdf import FPDF
import io
import os
import cv2
from datetime import datetime

def generate_pdf(filtered_history):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Добавляем шрифт
    pdf.add_font("DejaVu", "", "fonts/DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", "", 16)

    # Титульный лист
    pdf.cell(0, 10, "Отчёт по обработанным данным", ln=True, align="C")
    pdf.ln(10)

    # Собираем статистику
    total_cows = 0
    total_files = len(filtered_history)
    total_processing_time = 0
    total_file_size = 0
    min_cows = float('inf')
    max_cows = 0
    average_cows = 0

    # Подсчёт использованных моделей
    model_stats = {}

    # Добавляем общую статистику в начало отчёта
    pdf.ln(10)
    pdf.set_font("DejaVu", "", 14)
    pdf.cell(0, 10, "Общая статистика:", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("DejaVu", "", 12)
    for record in filtered_history:
        detections = record.get("detections", {})
        if record["type"] == "image":
            cows = detections.get('cows', 0)
        else:
            cows = detections.get('total_cows', 0)

        total_cows += cows
        total_processing_time += record.get('processing_time_sec', 0)
        total_file_size += record.get('file_size_kb', 0)

        min_cows = min(min_cows, cows)
        max_cows = max(max_cows, cows)

        # Подсчитываем, сколько раз использовалась каждая модель
        model_used = record.get('model_used', 'Unknown')
        model_stats[model_used] = model_stats.get(model_used, 0) + 1

    average_cows = total_cows / total_files if total_files > 0 else 0

    # Вывод общей статистики
    pdf.cell(0, 8, f"Общее количество коров: {total_cows}", ln=True, align="C")
    pdf.cell(0, 8, f"Минимальное количество коров в одной записи: {min_cows}", ln=True, align="C")
    pdf.cell(0, 8, f"Максимальное количество коров в одной записи: {max_cows}", ln=True, align="C")
    pdf.cell(0, 8, f"Среднее количество коров: {average_cows:.2f}", ln=True, align="C")
    pdf.cell(0, 8, f"Общее время обработки: {total_processing_time} сек", ln=True, align="C")
    pdf.cell(0, 8, f"Общий размер файлов: {round(total_file_size, 2)} КБ", ln=True, align="C")

    pdf.ln(15)  # Отступ перед статистикой по моделям

    # Вывод статистики по моделям
    pdf.set_font("DejaVu", "", 14)
    pdf.cell(0, 10, "Статистика по моделям:", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("DejaVu", "", 12)
    for model_name, count in model_stats.items():
        pdf.cell(0, 8, f"{model_name}: {count} обработанных файлов", ln=True, align="L")

    pdf.ln(10)  # Отступ перед основной частью отчёта

    # Теперь добавляем информацию о каждом файле (только обработанные файлы)
    for record in filtered_history:
        # Только обработанные файлы
        if not record.get('processed_filename'):
            continue

        pdf.set_font("DejaVu", "", 12)
        record_date = datetime.fromisoformat(record["timestamp"]).strftime("%d.%m.%Y %H:%M")
        pdf.cell(0, 8, f"Дата/время: {record_date}", ln=True)
        pdf.cell(0, 8, f"Тип файла: {record['type'].capitalize()}", ln=True)

        # Добавляем информацию о модели, которая использовалась для обработки файла
        model_used = record.get("model_used", "Не указана")
        pdf.cell(0, 8, f"Модель использованная: {model_used}", ln=True)

        detections = record.get("detections", {})
        if record["type"] == "image":
            det_text = f"Найдено коров: {detections.get('cows', 0)}"
        else:
            det_text = f"Сумма коров: {detections.get('total_cows', 0)}, Среднее на кадр: {detections.get('average_cows_per_frame', 0)}"
        pdf.cell(0, 8, f"Детекции: {det_text}", ln=True)

        pdf.cell(0, 8, f"Размер файла: {round(record.get('file_size_kb', 0), 2)} КБ", ln=True)
        pdf.cell(0, 8, f"Разрешение: {record.get('resolution', '-')}", ln=True)
        pdf.cell(0, 8, f"Время обработки: {record.get('processing_time_sec', '-')} сек", ln=True)

        pdf.ln(5)

        # Вставляем обработанные изображения
        if record["type"] == "video":
            try:
                # Чтение первого кадра из обработанного видео
                cap = cv2.VideoCapture(record['processed_filename'])
                ret, frame = cap.read()
                if ret:
                    # Сохраняем кадр во временный файл
                    tmp_path = "temp_frame.jpg"
                    cv2.imwrite(tmp_path, frame)
                    # Вставляем изображение в PDF
                    pdf.image(tmp_path, w=90)
                    os.remove(tmp_path)
                cap.release()
            except Exception as e:
                print(f"Ошибка вставки кадра из видео в PDF: {e}")
        else:
            try:
                pdf.image(record['processed_filename'], w=90)
            except Exception as e:
                print(f"Ошибка вставки изображения в PDF: {e}")

        pdf.ln(10)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

    # Генерация PDF в память
    buffer = io.BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer
