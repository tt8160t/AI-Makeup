import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import os
from tensorflow import keras

print(tf.__version__)


def load_model(modelname):
    new_model = tf.keras.models.load_model(modelname)
    return new_model

def from_region_to_preict(img,modelname):
    img = read_file()
    img = img.convert('L')  # Ensure the image is in gray format
    img = img.resize((128, 128))  # Resize image to a fixed size
    img_array = np.array(img)
    x_images = img_array.astype('float32') / 255.0
    x_images = x_images.reshape((1, 128, 128))

    new_model = load_model(modelname)
    encoded_imgs = new_model.encoder(x_images).numpy()
    result = new_model.decoder(encoded_imgs).numpy()
    result = (result * 255).astype(np.uint8)

    # Remove the batch dimension and reshape if necessary
    result = result.reshape((128, 128)) 
    # Convert to an image and save
    output_image = Image.fromarray(result)
    #output_image.save("predict_image.jpg")
    output_image_np = np.array(output_image)
    return output_image_np

def read_file(folder='C:\\Users\\User\\Desktop\\Tony', filename= 'region_image.jpg'):
    img_path = os.path.join(folder, filename)
    img = Image.open(img_path)
    return img

# if __name__ == '__main__':
#     folder = ''
#     filename = 'region_image.jpg'
#     output_path=''
#     output_filename= "predict_image.jpg"
#     modelname = 'saved_model'

#     img_path = os.path.join(folder, filename)
#     img = Image.open('C:\\Users\\User\\Desktop\\Tony\\region_image.jpg')


#     result = from_region_to_preict(img,modelname)
#     output_image = Image.fromarray(result)
#     output_image.save(output_filename)