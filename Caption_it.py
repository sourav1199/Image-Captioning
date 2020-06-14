import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import string
import json
from time import time
import pickle
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50, preprocess_input , decode_predictions
from keras.preprocessing import image
from keras.models import Model,load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense,Dropout, Embedding,LSTM
from keras.layers.merge import add

model = load_model("gdrive/My Drive/Image Caption2/model_weights with dropout 0.5/model9.h5")

model_temp = ResNet50(weights="imagenet",input_shape = (224,224,3))

model_resnet = Model(model_temp.input, model_temp.layers[-2].output)

def preprocess_img(img):
    img=image.load_img(img,target_size=(224,224))
    img=image.img_to_array(img)
    # img=np.extend_dims(img,axis=0)
    img=img.reshape(1,224,224,3)
    img=preprocess_input(img)
    return img

def encode_img(img):
    img=preprocess_img(img)
    feature_vector=model_resnet.predict(img)
    feature_vector=feature_vector.reshape((1,feature_vector.shape[1]))
    return feature_vector

with open("gdrive/My Drive/Image Caption2/word_to_idx.pkl",'rb') as f:
    word_to_idx = pickle.load(f)
with open("gdrive/My Drive/Image Caption2/idx_to_word.pkl",'rb') as f:
    idx_to_word = pickle.load(f)

def predict_caption(photo):
    in_text = 'start_seq'
    max_len = 38
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence],maxlen=max_len,padding='post')
        y_pred = model.predict([photo,sequence])
        y_pred = y_pred.argmax()
        word = idx_to_word[y_pred]
    # print(word)
        in_text += (' '+ word)
        if word =='endseq':
            break
    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption

def caption_this_image(image):
    enc = encode_img(image)
    caption = predict_caption(enc)
    return caption