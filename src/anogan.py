from __future__ import print_function

import os
import cv2
import math
import numpy as np

from keras.models import Model
from keras.layers import Input, Reshape, Dense, MaxPooling2D, Conv2D, Flatten
from keras.layers import Conv2DTranspose, LeakyReLU
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import backend as k_backend
from keras.utils.generic_utils import Progbar
from settings import GENERATOR_WEIGHT, DISCRIMINATOR_WEIGHT, IMAGE_RESULT_DIR, WEIGHTS_DIR


# combine images for visualization
def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:4]
    image = np.zeros((height * shape[0], width * shape[1], shape[2]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], :] = img[:, :, :]

    return image


# generator model define
def generator_model():
    inputs = Input((10,))
    fc1 = Dense(input_dim=10, units=128 * 8 * 8)(inputs)
    fc1 = BatchNormalization()(fc1)
    fc1 = LeakyReLU(0.2)(fc1)
    fc2 = Reshape((8, 8, 128), input_shape=(128 * 8 * 8,))(fc1)
    up1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(fc2)
    conv1 = Conv2D(64, (3, 3), padding='same')(up1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    up2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv1)
    conv2 = Conv2D(64, (3, 3), padding='same')(up2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    up3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv2)
    conv3 = Conv2D(64, (3, 3), padding='same')(up3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    up4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv3)
    conv4 = Conv2D(3, (3, 3), padding='same')(up4)

    outputs = Activation('tanh')(conv4)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


# discriminator model define
def discriminator_model():
    inputs = Input((128, 128, 3))
    conv1 = Conv2D(64, (3, 3), padding='same')(inputs)
    conv1 = LeakyReLU(0.2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
    conv2 = LeakyReLU(0.2)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    fc1 = Flatten()(pool2)
    fc1 = Dense(1)(fc1)
    outputs = Activation('sigmoid')(fc1)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


# d_on_g model for training generator
def generator_containing_discriminator(g, d):
    d.trainable = False
    gan_input = Input(shape=(10,))
    x = g(gan_input)
    gan_output = d(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    # gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan


def load_model():
    d = discriminator_model()

    g = generator_model()

    d_optim = Adam()
    g_optim = Adam(lr=0.00002)
    g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    d.load_weights(DISCRIMINATOR_WEIGHT)
    g.load_weights(GENERATOR_WEIGHT)
    return g, d


# train generator and discriminator
def train(batch_size, x_train):
    # model define
    d = discriminator_model()
    d.summary()
    g = generator_model()
    g.summary()
    d_on_g = generator_containing_discriminator(g, d)
    d_optim = Adam(lr=0.00002)
    g_optim = Adam(lr=0.00002)
    g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    loss_list = []
    n_iter = int(x_train.shape[0] / batch_size)
    g_loss = None
    d_loss = None
    for epoch in range(200):
        print("Epoch is", epoch)
        progress_bar = Progbar(target=n_iter)
        for index in range(n_iter):
            # create random noise -> U(0,1) 10 latent vectors
            noise = np.random.uniform(0, 1, size=(batch_size, 10))

            # load real data & generate fake data
            image_batch = x_train[index * batch_size:(index + 1) * batch_size]
            generated_images = g.predict(noise, verbose=0)

            # visualize training results
            if index % 100 == 0:
                image = combine_images(generated_images[0:9])
                image = image * 255
                image_path = os.path.join(IMAGE_RESULT_DIR, '{}_{}.png'.format(str(epoch), str(index)))
                cv2.imwrite(image_path, image[:, :, ::-1])

            # attach label for training discriminator
            x = np.concatenate((image_batch, generated_images))
            y = np.array([1] * batch_size + [0] * batch_size)

            # training discriminator
            d_loss = d.train_on_batch(x, y)

            # training generator
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, np.array([1] * batch_size))
            d.trainable = True

            progress_bar.update(index, values=[('g', g_loss), ('d', d_loss)])
        print('')
        loss_list.append([g_loss, d_loss])
        # save weights for each epoch
        g_path = os.path.join(WEIGHTS_DIR, 'generator_{}.h5'.format(str(epoch)))
        d_path = os.path.join(WEIGHTS_DIR, 'discriminator_{}.h5'.format(str(epoch)))
        g.save_weights(g_path, True)
        d.save_weights(d_path, True)

    return d, g, loss_list


# generate images
def generate(batch_size):
    g = generator_model()
    g.load_weights(GENERATOR_WEIGHT)
    noise = np.random.uniform(0, 1, (batch_size, 10))
    generated_images = g.predict(noise)
    return generated_images


# anomaly loss function
def sum_of_residual(y_true, y_pred):
    return k_backend.sum(k_backend.abs(y_true - y_pred))


# discriminator intermediate layer feautre extraction
def feature_extractor(d=None):
    if d is None:
        d = discriminator_model()
        d.load_weights(DISCRIMINATOR_WEIGHT)
    intermediate_model = Model(inputs=d.layers[0].input, outputs=d.layers[-7].output)
    intermediate_model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return intermediate_model


# anomaly detection model define
def anomaly_detector(g=None, d=None):
    if g is None:
        g = generator_model()
        g.load_weights(GENERATOR_WEIGHT)
    intermediate_model = feature_extractor(d)
    intermediate_model.trainable = False
    g = Model(inputs=g.layers[1].input, outputs=g.layers[-1].output)
    g.trainable = False
    # Input layer can't be trained. Add new layer as same size & same distribution
    a_input = Input(shape=(10,))
    g_input = Dense(10, trainable=True)(a_input)
    g_input = Activation('sigmoid')(g_input)

    # G & D feature
    g_out = g(g_input)
    d_out = intermediate_model(g_out)
    model = Model(inputs=a_input, outputs=[g_out, d_out])
    model.compile(loss=sum_of_residual, loss_weights=[0.90, 0.10], optimizer='rmsprop')

    # batch norm learning phase fixed (test) : make non trainable
    k_backend.set_learning_phase(0)

    return model


# anomaly detection
def compute_anomaly_score(model, x, intermediate_model, iterations=500):
    z = np.random.uniform(0, 1, size=(1, 10))

    #    intermediate_model = feature_extractor(d)
    d_x = intermediate_model.predict(x)

    # learning for changing latent
    loss = model.fit(z, [x, d_x], batch_size=1, epochs=iterations, verbose=0)
    similar_data, _ = model.predict(z)

    loss = loss.history['loss'][-1]

    return loss, similar_data
