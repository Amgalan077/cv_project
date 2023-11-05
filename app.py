import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from io import BytesIO
import requests
from tools_models.autoen import ConvAutoencoder


button_style = """
    <style>
    .center-align {
        display: flex;
        justify-content: center;
    }
    </style>
    """

DEVICE = 'cuda'
model = ConvAutoencoder().to(DEVICE)
model.load_state_dict(torch.load('tools_models\\autoend.pt'))

transform = transforms.Compose([
    transforms.ToTensor(),  # Преобразование изображения в тензор
])
model.eval()

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

model.to('cuda')

if 'image' in locals():
    st.image(image, caption="Uploaded image", use_column_width=True)

    bw_image = image.convert('L')

    image_tensor = transform(bw_image).unsqueeze(0)

    image_tensor = image_tensor.to('cuda')

    with torch.no_grad():
        output = model(image_tensor)

    output = transforms.ToPILImage()(output[0].cpu())

    if st.button("Clear", type="primary"):
        st.image(output, caption="Annotated Image", use_column_width=True)