import streamlit as st
import requests
import json
from PIL import Image
from io import BytesIO
import boto3
import botocore


def parse_lambda_response_payload(payload):
    if payload is not None:
        payload_as_str = payload.read().decode("utf-8")
        return json.loads(payload_as_str)
    else:
        return {}


def invoke_lambda(name, payload):
    config = botocore.config.Config(read_timeout=900, connect_timeout=900, retries={"max_attempts": 0})
    lambda_client = boto3.client(
        "lambda",
        region_name="us-east-2",
        config=config,
        aws_access_key_id=st.secrets["access_key"],
        aws_secret_access_key=st.secrets["secret_key"],
    )
    r = lambda_client.invoke(FunctionName=name, InvocationType="RequestResponse", Payload=json.dumps(payload))
    return parse_lambda_response_payload(r.get("Payload"))


@st.cache
def render_img_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img


# Visual App

st.markdown("# ShopNet by mustard")
img_url = st.text_input("Enter the image URL")
if img_url:
    image = render_img_url(img_url)
    st.image(image, use_column_width=True)
    preds = invoke_lambda(st.secrets["lambda_name"], {"product": {"image_url": img_url}})
    preds = json.loads(preds["body"])
    preds
