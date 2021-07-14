import os
import base64
import cv2
import imutils
import numpy as np
import pickle
from matplotlib import pyplot as plt

def read_img(file):
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = imutils.resize(image, width=600)
    return image

def draw_rectangle(image, face): 
    # Draw bounding box
    (start_x, start_y, end_x, end_y) = face["rect"]
    detection_rect_color_rgb = (255, 0, 0)
    cv2.rectangle(img = image, 
                  pt1 = (start_x, start_y), 
                  pt2 = (end_x, end_y), 
                  color = detection_rect_color_rgb, 
                  thickness = 2)

    # Draw name and distance
    text = "{}: {:.3f}".format(face["name"], face["distance"])
    y = start_y - 15 if start_y - 15 > 15 else start_y + 15
    text_color_rgb = (255, 0, 0)
    bg_color_rgb = (246, 246, 246)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, fontFace=font, fontScale=0.45, thickness=1)
    text_w, text_h = text_size
    cv2.rectangle(image, (start_x, y), (start_x + text_w, y + text_h), bg_color_rgb, -1)
    cv2.putText(img = image, text = text, 
                org = (start_x, y+text_h), 
                fontFace = font, 
                fontScale = 0.45, 
                color = text_color_rgb, 
                thickness = 1)

def draw_rectangles(image, faces):
    # Draw rectangle over detections
    if len(faces) == 0:
        num_faces = 0
    else:
        num_faces = len(faces)
        # Draw a rectangle
        for face in faces:
            draw_rectangle(image, face)
    return num_faces, image

def prepare_image(image):
    # Create string encoding of the image
    image_content = cv2.imencode('.jpg', image)[1].tostring()
    # Create base64 encoding of the string encoded image
    encoded_image = base64.encodestring(image_content)
    to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
    return to_send

def plot_image(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def get_folder_dir(folder_name):
    cur_dir = os.getcwd()
    folder_dir = cur_dir + "/" + folder_name + "/"
    return folder_dir

def load_dataset():
    f = open("models/face_dataset", "rb")
    dataframe = pickle.loads(f.read())
    dataframe.head()
    return dataframe

def save_dataset(dataframe):
    f = open("models/face_dataset", "wb")
    f.write(pickle.dumps(dataframe))
    f.close()