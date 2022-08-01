import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import tensorflow_hub as hub

# 모델경로
MODEL_PATH = "microsoft_cats_vs_dogs_model.h5"
cat_dog_model = load_model(
    MODEL_PATH,
    custom_objects={'KerasLayer': hub.KerasLayer}
)


def cat_dog_predict(file):
    print(file)
    image = load_img(file, target_size=(260, 260))
    input_arr = img_to_array(image)
    input_arr = np.array([input_arr])
    input_arr = input_arr.astype('float32')
    input_arr /= 255.0
    input_arr = input_arr.reshape(-1, 260, 260, 3)
    prediction = cat_dog_model.predict(input_arr)
    if prediction[0][0] < 0.5:
        return "Cat"
    return "Dog"