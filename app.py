import streamlit as st
import requests
import json
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import boto3
import botocore
import math

if st.session_state.get("all_classes") is None:
    with open("all_classes.txt", "r") as f:
        all_classes = f.readlines()
    st.session_state["all_classes"] = [x.strip() for x in all_classes]

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
def get_image(url):
    response = requests.get(url)
    img_content = BytesIO(response.content)
    return img_content


def resize_image(img, size):
    basewidth = size
    wpercent = basewidth / float(img.size[0])
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    return img


def draw_bounding_box_on_image(
    image, ymin, xmin, ymax, xmax, display_str, color="black", thickness=2, use_normalized_coordinates=False
):
    """Adds a bounding box to an image.
    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.
    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.
    If the top of the bounding box extends to the edge of the image, the strings
    are displayed below the bounding box.
    Args:
        image: a PIL.Image object.
        ymin: ymin of bounding box.
        xmin: xmin of bounding box.
        ymax: ymax of bounding box.
        xmax: xmax of bounding box.
        color: color to draw bounding box. Default is red.
        thickness: line thickness. Default value is 4.
        display_str_list: list of strings to display in box
                        (each to be shown on its own line).
        use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    if thickness > 0:
        draw.line(
            [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color
        )
    try:
        font = ImageFont.truetype("IBMPlexSans-Regular.ttf", int(0.03 * im_width))
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in [display_str]]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    text_width, text_height = font.getsize(display_str)
    margin = math.ceil(0.1 * text_height)
    draw.rectangle(
        [
            (left - 4 * margin, text_bottom - text_height - 2 * margin),
            (left + text_width + 4 * margin, text_bottom + 2 * margin),
        ],
        fill=color,
    )
    draw.text((left + margin, text_bottom - text_height - margin), display_str, fill="white", font=font)
    text_bottom -= text_height - 2 * margin


# Visual App
st.markdown("# ShopNet by mustard")
st.markdown("""Detects clothes and their types in images.""")
with st.beta_expander("Click to see all clothing types that are supported:"):
    st.session_state["all_classes"]
img_url = st.text_input("Enter the image URL")
if img_url:
    image = Image.open(get_image(img_url))
    w, h = image.size
    cx = w / 2
    with st.spinner("Processing the image"):
        preds = invoke_lambda(st.secrets["lambda_name"], {"product": {"image_url": img_url}})
    preds = json.loads(preds["body"])
    if len(preds) > 0:
        for pred in preds:
            x = (pred["xmin"] + pred["xmax"]) / 2
            y = (pred["ymin"] + pred["ymax"]) / 2
            r = int(0.01 * w)
            draw_bounding_box_on_image(image, pred["ymin"], pred["xmin"], pred["ymax"], pred["xmax"], pred["name"])
        st.image(image, use_column_width=True)
    else:
        st.markdown("No clothing objects found in the image.")
        st.image(image, use_column_width=True)
