import streamlit as st
import requests
import json
from PIL import Image
from io import BytesIO
import torch
import clip as clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

st.markdown("# Zero Shot Image Classifier")


@st.cache
def render_img_url(url, local=False):
    if not local:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(url)
    return img


def predict(image, text_classes):
    image = preprocess(image).unsqueeze(0).to(device)
    text = clip.tokenize(text_classes).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    preds = {
        text_classes[i]: "{0:.0%}".format(probs[0][i])
        for i in range(len(text_classes))
    }
    return preds


img_url = st.text_input('Enter the image URL')
if img_url:
    image = render_img_url(img_url)
    st.image(image, use_column_width=True)
    st.markdown("""
Enter comma separated text classes below. For example if you wanted to classify a picture of an animal, you could enter: 

`dog, cat, tiger, bear`
""")
    raw_text_labels = st.text_area("", value='', height=None, max_chars=None)
    if raw_text_labels:
        labels = [x.strip() for x in raw_text_labels.split(",")]
        preds = predict(image, labels)
        preds