import streamlit as st
import cv2
from io import BytesIO
import time
import albumentations as A
import numpy as np
import boto3

# Initialize the Boto3 S3 client
aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
aws_default_region = st.secrets["AWS_DEFAULT_REGION"]
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=aws_default_region)

# Specify the S3 bucket names
data_bucket = 'concretecracks-input'

picture = 0

data_resize = A.Resize(256, 256)

st.set_page_config(layout="wide", page_title="Concrete Crack Detection from Images")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.write("# Demo: Detecting Cracks on Concrete Surfaces")
st.write(
    "Try uploading an image of a concrete structural element. Image quality will be reduced automatically to (256x256) to speed up computations. This CV model is still under development."
)


# download
def convert_image(img):
    byte_im = img.getvalue()
    return byte_im

def get_np_array(input):
     return np.asarray(bytearray(input.read()), dtype=np.uint8)

def detect_cracks(upload, upload_img):
    input_resize = data_resize(image= upload_img)
    col1.write(":camera: Original Image")
    col1.image(input_resize['image'], use_column_width=True)
    s3.upload_fileobj(upload, Bucket=data_bucket, Key='images/streamlit-input')
    with st.spinner('Wait for it...'):
        time.sleep(15)
    st.success('Done!')
    #paginator = s3.get_paginator('list_objects_v2')
    #result_iterator = paginator.paginate(Bucket=data_bucket, Prefix='masked/')
    #png_files = [obj['Key'] for result in result_iterator for obj in result.get('Contents', []) if obj['Key'].endswith('.png')]
    #latest_file = max(png_files, key=lambda x: s3.head_object(Bucket=data_bucket, Key=x)['LastModified'])
    obj = s3.get_object(Bucket=data_bucket, Key='masked/imagesstreamlit-input.png') # key=latest_file
    image_bytes = obj['Body'].read()
    col2.write(":microscope: Masked Image")
    col2.image(BytesIO(image_bytes), use_column_width=True)
    st.sidebar.markdown("\n")
    col2.download_button("Download!", convert_image(BytesIO(image_bytes)), "masked.png", "image/png", use_container_width=True)

col1, col2 = st.columns(2)

st.sidebar.markdown("### Detecting Cracks on Concrete Surfaces")
st.sidebar.markdown(
"""
Model deployed on AWS Lambda


By [Zachary Hamida](https://zachamida.github.io).
"""
)
st.sidebar.markdown("---")
my_upload = st.sidebar.file_uploader("Upload an image of a concrete surface", type=["png", "jpg", "jpeg"])
st.sidebar.markdown("---")
cam = st.sidebar.checkbox('Use device camera to take a picture')
st.sidebar.markdown("---")

if cam:
    picture = st.camera_input("Take a picture")

if picture:
    thumbnail = get_np_array(picture)
    thumbnail = cv2.imdecode(thumbnail,0)
    detect_cracks(BytesIO(picture.getvalue()), upload_img=thumbnail)
elif my_upload is not None:
    thumbnail = get_np_array(my_upload)
    thumbnail = cv2.imdecode(thumbnail, 0)
    detect_cracks(BytesIO(my_upload.getvalue()), upload_img=thumbnail)
else:
    input = open("./example.jpg", 'rb')
    thumbnail = cv2.imread("./example.jpg")
    detect_cracks(upload=input, upload_img=thumbnail)