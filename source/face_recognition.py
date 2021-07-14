import cv2
from skimage import io
from PIL import Image

from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions

import tensorflow as tf
import scipy.spatial.distance as distance
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from source.utils import load_dataset, save_dataset


def save_img(filename, image):
    io.imsave(filename, image)

def detect_faces(image, min_confidence = 0.2):
    '''Detect face in an image'''
    detector = MTCNN()
    faces_list = detector.detect_faces(image)
    return faces_list

def faces_extract(image, required_size=(224, 224)):
    faces_pos = detect_faces(image)
    
    faces_arr = []
    for i in range(len(faces_pos)):
        # get coordinates
        x1, y1, width, height = faces_pos[i]['box']
        
        if (width < height):
            x1 = x1 - (height - width)//2
            width = height
        elif (width > height):
            y1 = y1 - (width - height)//2
            height = width
        x2, y2 = x1 + width, y1 + height

        # Calculating a center point of small face
        center = (width // 2, height // 2)

        # get keypoint (face landmarks)
        keypoints = faces_pos[i]['keypoints']
        left_eye_x, left_eye_y = keypoints['left_eye']
        right_eye_x, right_eye_y = keypoints['right_eye']

        # compute arctan
        delta_x = right_eye_x - left_eye_x
        delta_y = right_eye_y - left_eye_y
        angle = np.arctan(delta_y/delta_x)
        angle = (angle * 180) / np.pi

        # alingment face
        M = cv2.getRotationMatrix2D(center, (angle), 1.0)
        rotated = cv2.warpAffine(image[y1:y2, x1:x2], M, (width, height))

        # extract face
        face_img = Image.fromarray(rotated)
        face_img = face_img.resize(required_size)
        face_array = np.asarray(face_img)
        faces_arr.append(face_array)

    faces_list = [faces_pos, faces_arr]
    return faces_list

def get_embedding(faces_list):
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    
    samples = np.asarray(faces_list, 'float32')
    samples = preprocess_input(samples, version=2)

    embeddings = model.predict(samples)
    return embeddings

def face_match(dataframe, embedding, thresh=0.32):
    label = 'Unknow'
    
    # compute dítance to all point in dataframe
    distances = [distance.cosine(x, embedding) for x in dataframe['embedding']]
    # get three index smalless distance in list
    top_idxs = np.argsort(distances)[:3]
    index = Counter(top_idxs).most_common(1)[0][0]
    
    dis = distances[index]
    if (dis <= thresh):
        label = dataframe['label'][index]
    
    return (label, dis)

def faces_match(dataframe, list_embedding, thresh=0.32):
    y_pred = [face_match(dataframe, embedding, thresh) for embedding in list_embedding]
    return y_pred

def recognize_faces(image):
    # load dataset
    dataset = load_dataset()

    # extract face
    faces_pos, faces_arr = faces_extract(image)
    embeddings = get_embedding(faces_arr)

    # face recognition
    faces_list = []
    for i in range(len(faces_pos)):
        x1, y1, width, height = faces_pos[i]['box']
        x2, y2 = x1 + width, y1 + height
        label, dist = face_match(dataset, embeddings[i], thresh=0.32)

        face_dict = {}
        face_dict['rect'] = [x1, y1, x2, y2]
        face_dict['name'] = label
        face_dict['distance'] = dist
        faces_list.append(face_dict)

    return faces_list

def training_face(image, name):
    # load dataset
    dataset = load_dataset()

    # extract face
    faces_pos, faces_arr = faces_extract(image)
    embeddings = get_embedding(faces_arr)
    
    # Create train dataframe from embedding and name
    names = [name]
    list_tuples = list(zip(embeddings, names))  
    df_train = pd.DataFrame(list_tuples, columns=['embedding', 'label']) 
    dataset = pd.concat([dataset, df_train], ignore_index=True)
    
    x1, y1, width, height = faces_pos[0]['box']
    x2, y2 = x1 + width, y1 + height

    face_dict = {}
    face_dict['rect'] = [x1, y1, x2, y2]
    face_dict['name'] = name
    face_dict['distance'] = 0.
    
    # Save embedding
    save_dataset(dataset)
    return [face_dict]

