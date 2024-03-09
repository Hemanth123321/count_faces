import dlib
from PIL import Image, ImageDraw, ExifTags
import numpy as np
import streamlit as st

st.header("RCEE::AI&DS")
st.title("FACE RECOGNITION AND COUNT ")

uploaded_image = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())

        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass
    image_np = np.array(image)
    detector = dlib.get_frontal_face_detector()
    dets = detector(image_np, 1)
    draw = ImageDraw.Draw(image)
    for det in dets:
        top, right, bottom, left = det.top(), det.right(), det.bottom(), det.left()
        draw.rectangle(((left, top), (right, bottom)), outline="blue", width=3)
    st.image(image, caption='Uploaded Image with Face Recognition', use_column_width=True)
    num_faces = len(dets)
    st.write(f'Number of Faces Detected: {num_faces}')
