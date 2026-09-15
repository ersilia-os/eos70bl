
from keras.layers import Layer, MultiHeadAttention
from keras import activations,layers
import keras.backend as K
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras

class Length(Layer):
    def call(self, inputs, **kwargs):
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), -1) + tf.keras.backend.epsilon()) ###

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config

# @keras.saving.register_keras_serializable()
def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True) ###
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + tf.keras.backend.epsilon()) ###
    return scale * vectors


def PrimaryCap(inputs, dim_vector, n_channels, kernel_size, strides, padding):
    output = layers.Conv1D(filters=dim_vector * n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv1d')(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_vector], name='primarycap_reshape')(output)
    output_shape = outputs.shape[1:]
    return layers.Lambda(squash, output_shape=output_shape, name='primarycap_squash')(outputs)


# def PrimaryCap(inputs, dim_vector, n_channels, kernel_size, strides, padding):
#     # 进行卷积操作
#     output = layers.Conv1D(
#         filters=dim_vector * n_channels,
#         kernel_size=kernel_size,
#         strides=strides,
#         padding=padding,
#         name='primarycap_conv1d'
#     )(inputs)
#     # 进行形状重塑
#     outputs = layers.Reshape(
#         target_shape=[-1, dim_vector],
#         name='primarycap_reshape'
#     )(output)

#     # 自定义 Squash 层
#     class SquashLayer(layers.Layer):
#         def call(self, inputs):
#             s_squared_norm = tf.reduce_sum(tf.square(inputs), axis=-1, keepdims=True)
#             scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + tf.keras.backend.epsilon())
#             return scale * inputs

#     # 应用 Squash 层
#     squashed_output = SquashLayer(name='primarycap_squash')(outputs)

#     return squashed_output


#define our own softmax function instead of K.softmax
def softmax(x, axis=-1):
    ex = tf.exp(x - tf.reduce_max(x, axis=axis, keepdims=True))
    return ex/tf.reduce_sum(ex, axis=axis, keepdims=True) ###


#A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, share_weights=True, activation='squash', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            # u_hat_vecs = tf.nn.conv1d(u_vecs, self.W)
            u_hat_vecs = tf.nn.conv1d(u_vecs, self.W, stride=1, padding='VALID')
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = tf.shape(u_vecs)[0] ###
        input_num_capsule = tf.shape(u_vecs)[1] ###
        u_hat_vecs = tf.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = tf.transpose(u_hat_vecs, (0, 2, 1, 3))
        #final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = tf.zeros_like(u_hat_vecs[:,:,:,0]) #shape = [None, num_capsule, input_num_capsule] ###
        for i in range(self.routings):
            c = softmax(b, 1)
            # o = K.batch_dot(c, u_hat_vecs, [2, 2])
            o = tf.einsum('bin,binj->bij', c, u_hat_vecs)
            if K.backend() == 'theano':
                o = tf.reduce_sum(o, axis=1) ###
            if i < self.routings - 1:
                o = tf.nn.l2_normalize(o, -1)
                # b = K.batch_dot(o, u_hat_vecs, [2, 3])
                b = tf.einsum('bij,binj->bin', o, u_hat_vecs)
                if K.backend() == 'theano':
                    b = tf.reduce_sum(b, axis=1) ###

        return self.activation(o)

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        })
        return config

class PartitionPadding(layers.Layer):
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'batch_size': self.batch_size,
        })
        return config

    def call(self, inputs):

        atom_features, molecule_indicator = inputs

        # Obtain subgraphs
        atom_features_partitioned = tf.dynamic_partition(
            atom_features, molecule_indicator, self.batch_size
        )

        # Pad and stack subgraphs
        num_atoms = [tf.shape(f)[0] for f in atom_features_partitioned]
        max_num_atoms = tf.reduce_max(num_atoms)
        atom_features_stacked = tf.stack(
            [
                tf.pad(f, [(0, max_num_atoms - n), (0, 0)])
                for f, n in zip(atom_features_partitioned, num_atoms)
            ],
            axis=0,
        )

        # Remove empty subgraphs (usually for last batch in dataset)
        gather_indices = tf.where(tf.reduce_sum(atom_features_stacked, (1, 2)) != 0)
        gather_indices = tf.squeeze(gather_indices, axis=-1)
        return tf.gather(atom_features_stacked, gather_indices, axis=0)

# from keras_multi_head import MultiHeadAttention
# from keras.activations import *
class TransformerEncoderReadout(layers.Layer):
    def __init__(
        self, num_heads=8, embed_dim=64, dense_dim=512, batch_size=64, **kwargs
    ):
        super().__init__(**kwargs)

        self.partition_padding = PartitionPadding(batch_size)
        self.attention = layers.MultiHeadAttention(num_heads, embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.average_pooling = layers.GlobalAveragePooling1D()

    def get_config(self):
        config = super().get_config()
        return config


    def call(self, inputs):
        x = self.partition_padding(inputs)
        padding_mask = tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]
        attention_output = self.attention(x, x, attention_mask=padding_mask)
        proj_input = self.layernorm_1(x + attention_output)
        proj_output = self.layernorm_2(proj_input + self.dense_proj(proj_input))
        return self.average_pooling(proj_output)
