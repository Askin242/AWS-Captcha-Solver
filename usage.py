import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import time
import io

characters = ['a', 'b', 'c', 'd', 'e', 'f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']


char_to_num = layers.StringLookup(vocabulary=characters, mask_token=None)
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)
    results = results[0][0][:, :6]
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    if any(i for i in output_text if i not in characters):
        return output_text, 0
    return output_text, 1

prediction_model = keras.models.load_model('final.keras')

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((200, 70))
    img = np.array(img).astype('float32') / 255.0
    img = np.transpose(img, (1, 0, 2))
    img = np.expand_dims(img, axis=0)
    return img

def predict(image_path):
    img = preprocess_image(image_path)
    start = time.time()
    preds = prediction_model.predict(img, verbose=0)
    end = time.time()
    pred_texts, percent = decode_batch_predictions(preds)
    ms = int((end - start) * 1000)

    return pred_texts[0], ms

def preprocess_image_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((200, 70))
    img = np.array(img).astype('float32') / 255.0
    img = np.transpose(img, (1, 0, 2))
    img = np.expand_dims(img, axis=0)
    return img


def predict_byte(image_bytes):
    img = preprocess_image_bytes(image_bytes)
    start = time.time()
    preds = prediction_model.predict(img, verbose=0)
    end = time.time()
    pred_texts, percent = decode_batch_predictions(preds)
    ms = int((end - start) * 1000)

    return pred_texts[0], ms
