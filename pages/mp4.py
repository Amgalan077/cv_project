import streamlit as st
import numpy as np
from ultralytics import YOLO
import tempfile
import cv2
import supervision as sv
import shutil  # Импортируйте модуль shutil

# Загрузка модели YOLO
model = YOLO("tools_models\yolov8\\best.pt")

# Функция для обработки видео
@st.cache_data
def process_frame(frame: np.ndarray, _) -> np.ndarray:
    results = model(frame, imgsz=1280, conf=0.8)[0]

    if len(results) > 0:
        detections = sv.Detections.from_yolov8(results)

        box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)

        labels = [f"{model.names[class_id]} {confidence:0.8f}" for _, _, confidence, class_id, _ in detections]
        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

    return frame

# Главное приложение Streamlit
st.title("Обработка видео с YOLO")

# Загрузка видео из локального файла
uploaded_file = st.file_uploader("Загрузите видео (MP4)", type=["mp4"])
if uploaded_file is not None:
    st.video(uploaded_file)

    # Обработка видео и отображение результата
    if st.button("Обработать"):
        st.write("Идет обработка видео...")

        # Сохранение загруженного видео на диск
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())

        result_video_path = sv.process_video(source_path=temp_file.name, target_path=f"result.mp4", callback=process_frame)
        
        # Убедитесь, что result_video_path установлено перед перемещением файла
        if result_video_path:
            # Переместите обработанное видео в желаемое местоположение (текущий рабочий каталог)
            shutil.move(result_video_path, "result.mp4")
        
            st.video("result.mp4")  # Отображение обработанного видео
        
            # Завершение обработки после окончания видео
            st.write("Обработка завершена")
