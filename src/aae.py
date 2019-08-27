#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

"""First we load the mnist dataset. We only need the images (from both the train and test set)."""

from keras.datasets import mnist

(x_train, y_train), (_, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_train = 2*x_train - 1

"""#Build the graph
We build the GAN network. We make three Sequential models for the encoder, the decoder and the discriminator.
"""

from tensorflow.keras.optimizers import Adam

autoencoder_optimizer = Adam(4e-6)
discriminator_optimizer = Adam(8e-6)
generator_optimizer = Adam(8e-6)

dropout = 0.8
latent_num = 2
w_stddev = 0.002

def w_init(stddev):
    return keras.initializers.RandomNormal(mean=0.0, stddev=stddev)

def dense(units, **kargs):
    return keras.layers.Dense(
        units=units,
        kernel_initializer=w_init(w_stddev),
        bias_initializer=w_init(w_stddev),
        **kargs)
                       

# encoder 
encoder = keras.models.Sequential(name="encoder_net")
encoder.add(keras.layers.Flatten())
encoder.add(dense(units=1024, name="dL1"))
encoder.add(keras.layers.ReLU())
encoder.add(dense(512, name="dL2"))
encoder.add(keras.layers.ReLU())
encoder.add(dense(256, name="dL3"))
encoder.add(keras.layers.ReLU())
encoder.add(dense(latent_num, name="encoded"))

# decoder
decoder = keras.models.Sequential(name="decoder_net")
decoder.add(keras.layers.Input(shape=(latent_num,)))
decoder.add(dense(256, name="gL1"))
decoder.add(keras.layers.ReLU())
decoder.add(dense(512, name="gL2"))
decoder.add(keras.layers.ReLU())
decoder.add(dense(1024,  name="gL3"))
decoder.add(keras.layers.ReLU())
decoder.add(dense(28*28,  name="decoded"))
decoder.add(keras.layers.Reshape((28, 28), name="unflattened_decoded"))

# discriminator 
discriminator = keras.models.Sequential(name="discriminator_net")
discriminator.add(keras.layers.Input(shape=(latent_num,)))
discriminator.add(keras.layers.Dropout(dropout))
discriminator.add(dense(units=1024, name="dL1"))
discriminator.add(keras.layers.ReLU())
discriminator.add(keras.layers.Dropout(dropout))
discriminator.add(dense(512, name="dL2"))
discriminator.add(keras.layers.ReLU())
discriminator.add(keras.layers.Dropout(dropout))
discriminator.add(dense(256, name="dL3"))
discriminator.add(keras.layers.ReLU())
discriminator.add(keras.layers.Dropout(dropout))
discriminator.add(dense(1, activation="sigmoid", name="prob"))


# reconstruction model
encoder.trainable = True
decoder.trainable = True

ae_input = keras.layers.Input(shape=(28, 28))
encoded = encoder(ae_input)
decoded = decoder(encoded)

autoencoder_model = keras.models.Model(name="autoencoder",
    inputs=ae_input, outputs=decoded)
autoencoder_model.compile(optimizer=autoencoder_optimizer,
                            loss='mse')

# adversarial models
encoder.trainable = False
decoder.trainable = False
discriminator.trainable = True

discr_input = keras.layers.Input(shape=(latent_num,))

discriminator_model = keras.models.Model(name="discriminator",
    inputs=discr_input, outputs=discriminator(discr_input))
discriminator_model.compile(optimizer=discriminator_optimizer,
                            loss='binary_crossentropy')

encoder.trainable = True
discriminator.trainable = False

generated_latents = keras.layers.Input(shape=(latent_num,))
gen_probs = decoder(generated_latents)

generator_model = keras.models.Model(name="generator",
    inputs=generated_latents, outputs=discriminator(generated_latents))
generator_model.compile(optimizer=generator_optimizer,
                        loss='binary_crossentropy')
encoder.summary()
decoder.summary()
discriminator.summary()
autoencoder_model.summary()
discriminator_model.summary()
generator_model.summary()

num_epochs  = 300
batch_size = 150
epochs_to_show = 5
prior_std = 1.0

img_indices = np.arange(x_train.shape[0])
num_imgs = len(img_indices)
batch_num = num_imgs//batch_size

def noise(size):
    x = np.random.normal(0,1, [size, latent_num])
    return x

def get_test(dim):
    x = np.linspace(-10, 10, dim)
    X, Y = np.meshgrid(x, x)
    return np.vstack((X.ravel(), Y.ravel())).T
test_dim = 16
test = get_test(test_dim)

from IPython.display import clear_output
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import glob, io


fig = plt.figure(figsize=(8,6), constrained_layout=True)
gs = fig.add_gridspec(3, 4)

ax1 = fig.add_subplot(gs[0,:])
rlim = 4
rlossl, = ax1.plot(0,0, c="blue")
dlossl, = ax1.plot(0,0, c="red")
alossl, = ax1.plot(0,0, c="green")
ax1.set_xlim([0, num_epochs])
ax1.set_ylim([0, rlim])

ax2 = fig.add_subplot(gs[1:,:2], aspect="equal")
digits = ax2.imshow(np.zeros([2, 2]), vmin=-1, vmax=1)
ax2.set_axis_off()

ax3 = fig.add_subplot(gs[1:,2:], aspect="equal")
cols = plt.cm.hsv(np.linspace(0, 1, 10))
points = []
for k in range(10):
    points.append(ax3.scatter(0,0, color=cols[k], s=5, alpha=0.2))
ax3.set_xlim([-10, 10])
ax3.set_ylim([-10, 10])

rloss = []
dloss = []
aloss = []
count = 0
for epoch in range(num_epochs):
    np.random.shuffle(img_indices)
    train_imgs = x_train[img_indices,...]
    epoch_rloss = [] 
    epoch_dloss = [] 
    epoch_aloss = [] 
    for batch in range(batch_num):
        
        curr_rloss, curr_aloss, curr_dloss = [0, 0, 0]

        real_images = train_imgs[batch*batch_size:(batch + 1)*batch_size]
        
        discr_batch_size = batch_size//3
        gen_batch_size = batch_size//3
        ae_batch_size = batch_size - (discr_batch_size + gen_batch_size)
        
        autoencoder_images = real_images[:ae_batch_size]
        autoencoder_noise = 0.1*np.random.randn(*autoencoder_images.shape)

        encoder.trainable = True
        decoder.trainable = True
        curr_rloss = autoencoder_model.train_on_batch(
                autoencoder_images + autoencoder_noise,
                autoencoder_images)

        adversarial_images = real_images[ae_batch_size:(ae_batch_size + discr_batch_size)] 
        adversarial_latents = encoder.predict(adversarial_images)
        discriminator_latents = prior_std*np.random.randn(*adversarial_latents.shape) 
        
        encoder.trainable = False
        decoder.trainable = False
        discriminator.trainable = True
        curr_dloss = discriminator_model.train_on_batch(
                np.vstack((discriminator_latents, adversarial_latents)), 
                np.hstack((np.ones(discr_batch_size), np.zeros(discr_batch_size))))
        
        generator_images = real_images[discr_batch_size:(discr_batch_size + gen_batch_size)] 
        generator_latents = encoder.predict(generator_images)
        encoder.trainable = True
        discriminator.trainable = False
        curr_aloss = generator_model.train_on_batch(
            generator_latents, 
            np.ones(gen_batch_size))
        
        epoch_rloss.append(curr_rloss)
        epoch_dloss.append(curr_dloss)
        epoch_aloss.append(curr_aloss)
        
    rloss.append(np.mean(epoch_rloss))
    dloss.append(np.mean(epoch_dloss))
    aloss.append(np.mean(epoch_aloss))
    
    if epoch%epochs_to_show == 0 or epoch == (num_epochs - 1):

        clear_output()
        print("epoch: %-10d rloss: %-10.7f  dloss: %-10.7f aloss: %-10.7f" % (
            epoch, rloss[-1], dloss[-1], aloss[-1]))
        
        rlossl.set_data(np.arange(epoch+1), np.hstack(rloss)*rlim)
        dlossl.set_data(np.arange(epoch+1), dloss)
        alossl.set_data(np.arange(epoch+1), aloss)

        test_gen_batch = decoder.predict(test).reshape(test_dim, test_dim, 28,28)
        digits.set_data(np.vstack([ np.hstack([img for img in test_gen_row])
            for test_gen_row in test_gen_batch[::-1]]))

        test_latents = encoder.predict(x_train)

        for k in range(10):
            digit_latents = test_latents[np.where(y_train==k)[0]]
            points[k].set_offsets(digit_latents)

        # plt.show()
        fig.savefig("aae-%06d.png"%count)
        count += 1

        # Open all the frames
        image_names = sorted(glob.glob("aae*.png"))
        if len(image_names) > 1:
            images = []

            for img_name in image_names:
                with open(img_name, "rb") as f:
                    frame = Image.open(io.BytesIO(f.read()))
                images.append(frame)
            #save the frames as an animated GIF
            images[0].save('aae.gif',
                    save_all=True,
                    append_images=images[1:],
                    duration=500,
                    loop=0)
