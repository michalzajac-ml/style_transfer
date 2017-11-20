import numpy as np
import os
import scipy
import tensorflow as tf
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.preprocessing import image

from utils import make_list, postprocess, create_gif


def normalized_gram_matrix(x):
    shape = tf.shape(x)
    channels = shape[3]
    x = tf.reshape(x, [-1, channels])
    return tf.matmul(x, x, transpose_a=True) / tf.cast(tf.size(x), tf.float32)


def denoising_loss(img):
    return (tf.reduce_mean((img[1:, :, :] - img[:-1, :, :]) ** 2) +
            tf.reduce_mean((img[:, 1:, :] - img[:, :-1, :]) ** 2))


CONTENT_LAYERS = [
    'block4_conv2',
    'block5_conv2',
]
STYLE_LAYERS = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1',
]


class StyleTransfer:
    def __init__(self, content, style, style_weight, denoising_weight, learning_rate):
        # prepare VGG models
        self.vgg_content_shaped = VGG19(include_top=False, input_shape=content.shape)
        self.vgg_style_shaped = VGG19(include_top=False, input_shape=style.shape)

        content_model = Model(
            inputs=self.vgg_content_shaped.input,
            outputs=[self.vgg_content_shaped.get_layer(layer).output for layer in CONTENT_LAYERS]
        )
        style_model_style_shaped = Model(
            inputs=self.vgg_style_shaped.input,
            outputs=[self.vgg_style_shaped.get_layer(layer).output for layer in STYLE_LAYERS]
        )
        style_model_content_shaped = Model(
            inputs=self.vgg_content_shaped.input,
            outputs=[self.vgg_content_shaped.get_layer(layer).output for layer in STYLE_LAYERS]
        )

        content = tf.constant(preprocess_input(content))
        style = tf.constant(preprocess_input(style))

        self.image = tf.Variable(tf.truncated_normal(tf.shape(content)))

        # content loss
        content_activations_original = make_list(content_model(tf.expand_dims(content, axis=0)))
        content_activations_created = make_list(content_model(tf.expand_dims(self.image, axis=0)))
        self.content_loss = 0
        for original, created in zip(content_activations_original, content_activations_created):
            self.content_loss += tf.reduce_mean((original - created) ** 2)
        self.content_loss /= len(CONTENT_LAYERS)

        # style loss
        style_activations_original = make_list(style_model_style_shaped(tf.expand_dims(style, axis=0)))
        style_activations_created = make_list(style_model_content_shaped(tf.expand_dims(self.image, axis=0)))
        self.style_loss = 0
        for original, created in zip(style_activations_original, style_activations_created):
            self.style_loss += tf.reduce_mean(
                (normalized_gram_matrix(original) - normalized_gram_matrix(created)) ** 2)
        self.style_loss /= len(style_activations_original)
        self.style_loss *= style_weight

        # total variation denoising loss
        self.denoising_loss = denoising_weight * denoising_loss(self.image)

        self.loss = self.content_loss + self.style_loss + self.denoising_loss

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_step = optimizer.minimize(self.loss, var_list=[self.image])

    def generate_image(self, num_steps, output_file, gif_file, save_every_n):
        with tf.Session() as session:
            gif_frames = []

            session.run(tf.global_variables_initializer())
            pretrained_vgg_weights = VGG19(include_top=False, weights='imagenet').get_weights()
            self.vgg_content_shaped.set_weights(pretrained_vgg_weights)
            self.vgg_style_shaped.set_weights(pretrained_vgg_weights)

            if save_every_n is not None:
                intermediate_results_dir = os.path.splitext(output_file)[0] + '_intermediate'
                if not os.path.exists(intermediate_results_dir):
                    os.makedirs(intermediate_results_dir)

            for i in range(1, num_steps + 1):
                _, image, content_loss, style_loss, denoising_loss = session.run([
                    self.train_step,
                    self.image,
                    self.content_loss,
                    self.style_loss,
                    self.denoising_loss
                ])
                print('iter %s: content loss %s, style loss %s, denoising loss %s' % (
                    i, content_loss, style_loss, denoising_loss))

                image = postprocess(image)
                image = np.clip(image, 0, 255).astype(np.uint8)

                if save_every_n is not None and i % save_every_n == 0:
                    file = os.path.join(intermediate_results_dir, 'iter_%d.png' % i)
                    print('Saving intermediate result to %s' % file)
                    scipy.misc.imsave(file, image)

                if gif_file is not None and i % 10 == 0:
                    gif_frames.append(image)

            scipy.misc.imsave(output_file, image)
            if gif_file is not None:
                create_gif(gif_frames, gif_file)
