import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

st.sidebar.header("Home page")
st.title(' SIMS segmentation ')
image_source = st.radio("Выберите источник изображения:", ("Файл", "URL"))
if image_source == "Файл":
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
else:
    url = st.text_input("Введите URL изображения...")
    if url:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
if 'image' in locals():
    im = np.array(image)
    # Используем дефолтный конфиг
    cfg = get_cfg()
    classes = ['lion',
 'giraffe',
 'hippopotamus',
 'lemur',
 'lion',
 'monkey',
 'penguin',
 'zebra']
    class_id_to_name = {i + 1: class_name for i, class_name in enumerate(classes)}
    yaml_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml" # путь к виду модели из библиотеки detectron2
    model_weights = 'tools_models\madagascar_weights.pth'   # путь к весам обученной нами модели
    cfg.merge_from_file(model_zoo.get_config_file(yaml_path))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.MODEL.WEIGHTS = model_weights  # Замените на путь к вашим весам модели
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
    cfg.MODEL.DEVICE = "cpu"
    # Создаем объект предиктора
    predictor = DefaultPredictor(cfg)
    # Выполняем сегментацию
    outputs = predictor(im[:, :, ::-1])
    # # Загрузка JSON-файла с метаданными
    # with open("./models/_annotations.coco.json", "r") as json_file:
    #     metadata = json.load(json_file)
    # Отображаем исходное изображение
    st.image(image, caption='Загруженное изображение.')
    # Визуализация результата сегментации
    v = Visualizer(im[:, :, ::-1], scale=0.5, instance_mode = ColorMode.SEGMENTATION)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    st.image(out.get_image()[:, :, ::-1], caption='Результат сегментации')