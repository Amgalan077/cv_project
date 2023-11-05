import streamlit as st
# from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
import torch

button_style = """
    <style>
    .center-align {
        display: flex;
        justify-content: center;
    }
    </style>
    """

CFG_MODEL_PATH = 'tools_models\minecraft\\best.pt'

@st.cache_resource

def load_yolov5_model(model_path):
    model = torch.hub.load('tools_models\minecraft\yolov5', 'custom',
                           path=model_path, source='local', force_reload=True)
    return model

model = load_yolov5_model(CFG_MODEL_PATH)

image_source = st.radio("Choose the option of uploading the image of tumor:", ("File", "URL"))

if image_source == "File":
    uploaded_file = st.file_uploader("Upload the image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
else:
    url = st.text_input("Enter the URL of image...")
    if url:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))

st.markdown(button_style, unsafe_allow_html=True)

if 'image' in locals():
    st.image(image, caption="Uploaded image", use_column_width=True)
    annotated_image = model(image).render()
    if st.button("Detect", type="primary"):
        st.image(annotated_image, caption="Annotated Image", use_column_width=True)