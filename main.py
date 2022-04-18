import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers, Model
from sklearn.model_selection import train_test_split

import numpy as np
from keras import Model
from keras.layers import Conv2D, PReLU,BatchNormalization, Flatten
from keras.layers import UpSampling2D, LeakyReLU, Dense, Input, add
from tqdm import tqdm


#code for building generator


def res_block(input_dim):
    model = Conv2D(64, (3,3), padding = 'same' )(input_dim)
    model = BatchNormalization()(model)
    model = PReLU(shared_axes = [1,2])(model)
    model = Conv2D(64, (3,3), padding = 'same' )(model)
    model = BatchNormalization()(model)
    return add([input_dim, model])
def upscale_block(input_dim):
    model = Conv2D(256,(3,3), strides=1, padding = 'same')(input_dim)
    model = UpSampling2D(size = (2,2))(model)
    model = PReLU(shared_axes=[1, 2])(model)
    return model
def generator(input, res_range,upscale_range):
    model = Conv2D(64,(9,9), strides=1, padding = 'same')(input)
    model = PReLU(shared_axes = [1,2])(model)
    model1 = model
    for i in range(res_range):
        model = res_block(model)
    model = Conv2D(64, (3,3), padding = 'same' )(model)
    model = BatchNormalization()(model)
    model = add([model,model1])
    for i in range(upscale_range):
        model  =upscale_block(model)
    out = Conv2D(3, (9,9), stride = 1, padding='same')(model)
    return Model(model, out)


#code for building discriminator
def discrim_block(input_dim, fmaps = 64, strides = 1):
    model = Conv2D(fmaps, (3,3), padding = 'same', strides  = strides)(input_dim)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    return model
def discriminator(input):
    model = Conv2D(64,(3,3),padding='same')(input)
    model = LeakyReLU()(model)
    model = discrim_block(model, strides = 2)
    model = discrim_block(model, fmaps  = 128)
    model = discrim_block(model, fmaps = 128, strides = 2)
    model = discrim_block(model, fmaps=256)
    model = discrim_block(model, fmaps=256, strides=2)
    model = discrim_block(model, fmaps=512)
    model = discrim_block(model, fmaps=512, strides=2)
    model = Dense(1024, activation = LeakyReLU)
    out = Dense(1, activation= 'binary_crossentropy')
    return Model(model, out)

#introducing vgg19 layer
from keras.applications import VGG19


def build_vgg(hr_shape):
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=hr_shape)

    return Model(inputs=vgg.inputs, outputs=vgg.layers[10].output)


# Combined model
def create_comb(gen_model, disc_model, vgg, lr_ip, hr_ip):
    gen_img = gen_model(lr_ip)

    gen_features = vgg(gen_img)

    disc_model.trainable = False
    validity = disc_model(gen_img)

    return Model(inputs=[lr_ip, hr_ip], outputs=[validity, gen_features])

#defining losses
