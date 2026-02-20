import csv

import cv2
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
from tensorflow.keras.layers import Conv2D, MaxPooling2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, PReLU, Input, \
    BatchNormalization, GlobalMaxPooling2D, SeparableConv2D, LeakyReLU, Concatenate,Lambda,Flatten,SeparableConvolution2D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers, layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping
from tensorflow.keras.layers import Activation, Dropout, Flatten, AveragePooling2D, add
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import regularizers
import os
import numpy as np
import warnings
import time
import tensorflow.keras.layers
from tensorflow.python.keras.applications.efficientnet import EfficientNetB7
import sklearn
import seaborn as sns
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import concatenate, Conv2DTranspose, ZeroPadding2D, multiply, UpSampling2D, Add, \
    ReLU, MaxPool2D, Reshape, GlobalAvgPool2D, Multiply
from tensorflow.python.keras.regularizers import l2
from tensorflow.keras import layers as L

import cbam
import numpy as np
import uuid
# 用于避免卷积层同名报错
unique_random_number = uuid.uuid4()
# import keras_cv_attention_models.fastervit
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# 超参数
# -------------------------------------------------

class Metrics(Callback):

    aa=0
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs=None):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict,average='macro')
        if logs is not None:
            val_accuracy = logs.get('val_accuracy')
            if val_accuracy is not None and val_accuracy>=0.9566 and val_accuracy>Metrics.aa:
                # print(f"Epoch {epoch + 1}: Validation Accuracy = {val_accuracy:.4f}")
                fer_json = self.model.to_json()
                with open("bestParam.json", "w") as json_file:
                    json_file.write(fer_json)
                self.model.save_weights("bestParam.h5")
                Metrics.aa = val_accuracy
                print(Metrics.aa)
                print("Saved model to disk")

        _val_recall = recall_score(val_targ, val_predict,average='macro')
        _val_precision = precision_score(val_targ, val_predict,average='macro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print (" — val_f1: % f — val_precision: % f — val_recall % f" % (_val_f1, _val_precision, _val_recall))
        return

import cbam
import numpy as np
import uuid
# 用于避免卷积层同名报错
unique_random_number = uuid.uuid4()

from tensorflow import keras
def swin_mlp_block(x, dim, window_size=7, mlp_ratio=4):
    """Window-MLP block: LayerNorm → Window partition → MLP → reverse window → Add"""
    B, H, W, C = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    shortcut = x
    x = layers.LayerNormalization(epsilon=1e-5)(x)

    # Pad to multiple of window_size
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    x = layers.ZeroPadding2D(((0, pad_h), (0, pad_w)))(x)
    _, Hp, Wp, _ = x.shape

    # Reshape into windows (B, Nw, window_size, window_size, C)
    x = layers.Reshape((Hp // window_size, window_size,
                        Wp // window_size, window_size, C))(x)
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])  # (B, H/w, W/w, w, w, C)
    x = layers.Reshape((-1, window_size * window_size, C))(x)  # (B*Nw, w*w, C)

    # MLP along spatial dimension
    x = layers.Dense(window_size * window_size, activation='gelu')(x)
    x = layers.Dense(C)(x)

    # Restore spatial shape
    x = layers.Reshape((Hp // window_size, Wp // window_size,
                        window_size, window_size, C))(x)
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    x = layers.Reshape((Hp, Wp, C))(x)
    # Remove padding
    if pad_h > 0 or pad_w > 0:
        x = layers.Cropping2D(((0, pad_h), (0, pad_w)))(x)

    # Channel MLP
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    x = layers.Dense(int(dim * mlp_ratio), activation='gelu')(x)
    x = layers.Dense(dim)(x)
    return layers.Add()([shortcut, x])
# 1. 超参数
IMG_SIZE   = 224          # CIFAR-10 可设 32；若 224 会自动 resize
BATCH_SIZE = 128
EPOCHS     = 200
NUM_CLASS  = 10
MIXUP_ALPHA= 0.2
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
def mobilenet_block(x, filters, kernel_size, strides=1):
    """
    MobileNet Block: Depthwise Convolution + Pointwise Convolution
    """
    input_channels = x.shape[-1]
    # x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding='same', use_bias=False)(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.ReLU()(x)
    # x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3,  use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = layers.Conv2D(filters, 5,use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = layers.Conv2D(filters, 7, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    return x

def convnext_block(x, filters):
    """
    ConvNeXt Block: Depthwise Convolution + LayerNorm + MLP
    """
    # Ensure the number of groups matches the number of input channels
    input_channels = x.shape[-1]
    x = layers.Conv2D(input_channels, kernel_size=7, padding='same', groups=input_channels)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(4 * filters, activation='gelu')(x)
    x = layers.Dense(filters)(x)
    return x

def efficientnet_block(x, filters, kernel_size, strides=1, expand_ratio=6):
    """
    EfficientNet Block: MBConv Block with Swish activation
    """
    input_channels = x.shape[-1]
    expanded_channels = input_channels * expand_ratio
    if expand_ratio > 1:
        x = layers.Conv2D(expanded_channels, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

    x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if strides == 1 and input_channels == filters:
        x = layers.Add()([x, layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)])
    return x

IMAGE_ORDERING = 'channels_last'
class FerModel(object):
    def __init__(self):
        self.x_shape = (224, 224,3)
        self.epoch = 100
        self.batchsize = 16
        self.weight_decay = 0.0005
        self.classes = 10
        self.model = self.build_model()
        self.call_backs = self.get_call_backs()
        start = time.time()
        self.history = self.train()
        end = time.time()
        print(f'训练共耗时{round(end - start, 2)}s')

        # self.show_history()

    @staticmethod
    def get_call_backs():
        call_backs = [
            # ModelCheckpoint('./logs/' + 'best.h5',
            #                 save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2),
            TensorBoard('./logs/balanced_data_model'),
            # EarlyStopping(monitor='val_loss', patience=40)
        ]

        print("调用回调函数!!!")

        return call_backs

    # def build_model(self):
    #     # 骨干网络（Backbone）
    #     base_model = tensorflow.keras.applications.mobilenet.MobileNet(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    #
    #
    #     # 提取特征层（根据MobileNet结构选择）
    #     c3 = base_model.get_layer('block_6_expand_relu').output  # 1/8尺度
    #     c4 = base_model.get_layer('block_13_expand_relu').output  # 1/16尺度
    #     c5 = base_model.output  # 1/32尺度
    #
    #     # FPN构建
    #     # 顶层处理路径（P5）
    #     p5 = Conv2D(256, 1, name='c5_reduced')(c5)
    #     p5_upsampled = UpSampling2D()(p5)
    #
    #     # 中间层处理路径（P4）
    #     c4_reduced = Conv2D(256, 1, name='c4_reduced')(c4)
    #     p4 = Add()([p5_upsampled, c4_reduced])
    #     p4 = Conv2D(256, 3, padding='same', activation='relu')(p4)
    #     p4_upsampled = UpSampling2D()(p4)
    #
    #     # 底层处理路径（P3）
    #     c3_reduced = Conv2D(256, 1, name='c3_reduced')(c3)
    #     p3 = Add()([p4_upsampled, c3_reduced])
    #     p3 = Conv2D(256, 3, padding='same', activation='relu')(p3)
    #
    #     # 附加金字塔层
    #     p6 = Conv2D(256, 3, strides=2, padding='same', name='p6')(c5)
    #     p7 = Conv2D(256, 3, strides=2, padding='same', activation='relu', name='p7')(p6)
    #
    #     # 分类头（示例）
    #     outputs = Dense(10, activation='softmax')(x)
    #
    #     return Model(inputs=base_model.input, outputs=outputs)

    def ca_block(self,input_feature, ratio=16, name=""):
        channel = input_feature.shape[-1]
        h = input_feature.shape[1]
        w = input_feature.shape[2]

        x_h = Lambda(lambda x: K.mean(x, axis=2, keepdims=True))(input_feature)
        x_h = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1, 3]))(x_h)
        x_w = Lambda(lambda x: K.max(x, axis=1, keepdims=True))(input_feature)

        x_cat_conv_relu = Concatenate(axis=2)([x_w, x_h])
        x_cat_conv_relu = Conv2D(channel // ratio, kernel_size=1, strides=1, use_bias=False,
                                 name="ca_block_conv1_" + str(name))(x_cat_conv_relu)
        x_cat_conv_relu = BatchNormalization(name="ca_block_bn_" + str(name))(x_cat_conv_relu)
        x_cat_conv_relu = Activation('relu')(x_cat_conv_relu)

        x_cat_conv_split_h, x_cat_conv_split_w = Lambda(lambda x: tf.split(x, num_or_size_splits=[h, w], axis=2))(
            x_cat_conv_relu)
        x_cat_conv_split_h = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1, 3]))(x_cat_conv_split_h)
        x_cat_conv_split_h = Conv2D(channel, kernel_size=1, strides=1, use_bias=False,
                                    name="ca_block_conv2_" + str(name))(x_cat_conv_split_h)
        x_cat_conv_split_h = Activation('sigmoid')(x_cat_conv_split_h)

        x_cat_conv_split_w = Conv2D(channel, kernel_size=1, strides=1, use_bias=False,
                                    name="ca_block_conv3_" + str(name))(x_cat_conv_split_w)
        x_cat_conv_split_w = Activation('sigmoid')(x_cat_conv_split_w)

        output = multiply([input_feature, x_cat_conv_split_h])
        output = multiply([output, x_cat_conv_split_w])
        return output

    def SKBlock1(self,input_tensor, filters):
        # 分支1: 3x3卷积
        # branch1 = SeparableConv2D(filters, 3, padding='same', activation='relu')(input_tensor)

        branch2 = SeparableConv2D(filters, 5, padding='same', activation='relu')(input_tensor)
        # 分支2: 5x5卷积（等效两个3x3）
        branch3 = SeparableConv2D(filters, 7, padding='same', activation='relu')(input_tensor)

        # 注意力融合
        merged = Concatenate()([branch2,branch3])
        gap = GlobalAvgPool2D()(merged)
        fc = Dense(filters//8, activation='relu')(gap)
        attn = Dense(filters * 2, activation='softmax')(fc)  # 动态权重
        attn_b2, attn_b3 = tf.split(attn, num_or_size_splits=2, axis=1)
        output = Multiply()([branch2, attn_b2])+Multiply()([branch3, attn_b3])

        # 残差连接
        shortcut = Conv2D(filters, 1)(input_tensor) if input_tensor.shape[-1] != filters else input_tensor
        return tf.keras.layers.Add()([output, shortcut])

    # def build_model(self):
    #
    #     # # h0 = keras.Input(shape=(224, 224, 3),name="name='input_a")
    #     # # h1 = keras.Input(shape=(224, 224, 3),name="name='input_b")
    #     # #
    #     # # base_model = tensorflow.keras.applications.MobileNet(include_top=False, weights='imagenet', input_tensor=h0)
    #     # # # base = base_model.output
    #     # # base = base_model(h0)
    #     # #
    #     # # h1 = Conv2D(32, (3, 3), strides=2, padding="same")(h1)
    #     # # h2 = BatchNormalization()(h1)
    #     # # h3 = Activation('relu')(h2)
    #     # #
    #     # # h4 = self.ca_block(h3,name=str(uuid.uuid4()))
    #     # # h3 = tf.keras.layers.concatenate([h4, h3], name=str(uuid.uuid4()))
    #     # #
        # h4 = self.depth_point_conv2d(h3, s=[1, 1, 2, 1], channel=[64, 128])
    #     # # h5 = self.depth_point_conv2d(h4, s=[1, 1, 2, 1], channel=[128, 256])
    #     # # h6 = self.depth_point_conv2d(h5, s=[1, 1, 2, 1], channel=[256, 512])
    #     # # h7 = self.repeat_conv(h6)
    #     # # h8 = self.repeat_conv(h7)
    #     # # h9 = self.repeat_conv(h8)
    #     # # h10 = self.repeat_conv(h9)
    #     # #
    #     # # # h4 = self.ca_block(h10, name=str(uuid.uuid4()))
    #     # # # h10 = tf.keras.layers.concatenate([h4, h10], name=str(uuid.uuid4()))
    #     # #
    #     # # h11 = self.depth_point_conv2d(h10, s=[1, 1, 2, 1], channel=[512, 1024])
    #     # # h12 = self.repeat_conv(h11, channel=1024)
    #     # #
    #     # # h12 = GlobalAveragePooling2D()(h12)
    #     # # base = GlobalAveragePooling2D()(base)
    #     # #
    #     # # excitation = Dense(units=1024 // 16)(h12)
    #     # # excitation = Activation('relu')(excitation)
    #     # # excitation = Dense(units=1024)(excitation)
    #     # # excitation = Activation('sigmoid')(excitation)
    #     # #
    #     # # scale = multiply([excitation, base])
    #     # # xxx = multiply([base, h12])
    #     # #
    #     # # x = concatenate([xxx,base,scale])
    #     # # predictions = Dense(10, activation="softmax", kernel_initializer='he_normal')(x)
    #     # # # model = Model(inputs=[base_model.input,h1], outputs=predictions)
    #     # # model = keras.Model(inputs=[base_model.input,h1], outputs=predictions)
    #     #
    #     # inp1 = Input(shape=(128, 128, 1), name="img")  # 第一个输入，图像,名字为'img'
    #     # inp2 = Input(shape=(224, 224, 3), name="rate")  # 第二个输入，条件，名字为'rate'
    #     #
    #     # base_model = tensorflow.keras.applications.MobileNet(include_top=False, weights='imagenet',
    #     #                                                      input_shape=(224, 224, 3))
    #     # base = base_model(inp2)
    #     # base = self.py_block(base, 128, 256)
    #     #
    #     # x1 = Conv2D(32, 3, padding='same', activation="relu",
    #     #             # depthwise_regularizer=regularizers.l2(self.weight_decay),
    #     #             # pointwise_regularizer=regularizers.l2(self.weight_decay),
    #     #             kernel_initializer='he_normal', name='conv2d_1_1')(inp1)
    #     # x = MaxPooling2D()(x1)
    #     # x = BatchNormalization()(x)
    #     # x2 = SeparableConv2D(64, 5, padding='same', activation="relu",
    #     #                      # depthwise_regularizer=regularizers.l2(self.weight_decay),
    #     #                      # pointwise_regularizer=regularizers.l2(self.weight_decay),
    #     #                      kernel_initializer='he_normal', name='conv2d_1_2')(x)
    #     # x = MaxPooling2D()(x2)
    #     # x = BatchNormalization()(x)
    #     # x3 = SeparableConv2D(128, 7, padding='same', activation="relu",
    #     #                      # depthwise_regularizer=regularizers.l2(self.weight_decay),
    #     #                      # pointwise_regularizer=regularizers.l2(self.weight_decay),
    #     #                      kernel_initializer='he_normal', name='conv2d_1_3')(x)
    #     # x = MaxPooling2D()(x3)
    #     # x = BatchNormalization()(x)
    #     # x4 = SeparableConv2D(256, 9, padding='same', activation="relu",
    #     #                      # depthwise_regularizer=regularizers.l2(self.weight_decay),
    #     #                      # pointwise_regularizer=regularizers.l2(self.weight_decay),
    #     #                      kernel_initializer='he_normal', name='conv2d_1_4')(x)
    #     # x = MaxPooling2D()(x4)
    #     # x = BatchNormalization()(x)
    #     # x = Dropout(0.3)(x)
    #     #
    #     # base = GlobalAveragePooling2D()(base)
    #     # x = GlobalMaxPooling2D()(x)
    #     # x = tf.keras.layers.multiply([base, x])
    #     # # base = GlobalAveragePooling2D()(base)
    #     #
    #     # # excitation = Dense(units=256)(base)
    #     # # excitation = Activation('relu')(excitation)
    #     # # excitation = Dense(units=1024)(base)
    #     # # excitation = Activation('sigmoid')(excitation)
    #     #
    #     # # x = cbam.cbam_module(x)
    #     #
    #     # # base = GlobalAveragePooling2D(name=str(uuid.uuid4()))(base)
    #     # # print(base.shape[-1])
    #     # # x2 = Dense(1024,activation="relu")(x)
    #     # # x3 = GlobalMaxPooling2D(name=str(uuid.uuid4()))(x8)
    #     #
    #     # # x = tf.keras.layers.concatenate([x4,x5,x3], name=str(uuid.uuid4()))
    #     # # x = Dense(128, activation='relu')(x)
    #     # # excitation = Dense(units=256)(x)
    #     # # excitation = Activation('relu')(excitation)
    #     # # excitation = Dense(units=1024)(excitation)
    #     # # excitation = Activation('sigmoid')(excitation)
    #     #
    #     # # x1 = tf.keras.layers.multiply([x2, base], name=str(uuid.uuid4()))
    #     # #         #
    #     # #         # x = concatenate([base, x1,x])
    #     #
    #     # predictions = Dense(10, activation="softmax", kernel_initializer='he_normal')(x)
    #     # model = Model(inputs=[inp1, inp2], outputs=predictions)  # inp1和inp2作为输入，输出为out的模型实例化
    #     # model.summary()
    #     # return model
    #     h0 = keras.Input(shape=(224, 224, 3), name="name='input_a")
    #
    #     base_model = tensorflow.keras.applications.MobileNet(include_top=False, weights='imagenet', input_tensor=h0)
    #     # base = base_model.output
    #     base = base_model(h0)
    #
    #     h1 = Conv2D(32, (3, 3), strides=2, padding="same")(h0)
    #     h2 = BatchNormalization()(h1)
    #     h3 = Activation('relu')(h2)
    #
    #     h4 = self.ca_block(h3, name=str(uuid.uuid4()))
    #     h3 = tf.keras.layers.concatenate([h4, h3], name=str(uuid.uuid4()))
    #
    #     h4 = self.depth_point_conv2d(h3, s=[1, 1, 2, 1], channel=[64, 128])
    #     h5 = self.depth_point_conv2d(h4, s=[1, 1, 2, 1], channel=[128, 256])
    #     h6 = self.depth_point_conv2d(h5, s=[1, 1, 2, 1], channel=[256, 512])
    #     h7 = self.repeat_conv(h6)
    #     h8 = self.repeat_conv(h7)
    #     h9 = self.repeat_conv(h8)
    #     h10 = self.repeat_conv(h9)
    #
    #     h4 = self.ca_block(h10, name=str(uuid.uuid4()))
    #     h10 = tf.keras.layers.concatenate([h4, h10], name=str(uuid.uuid4()))
    #
    #     h11 = self.depth_point_conv2d(h10, s=[1, 1, 2, 1], channel=[512, 1024])
    #     h12 = self.repeat_conv(h11, channel=1024)
    #
    #     h12 = GlobalAveragePooling2D()(h12)
    #     base = GlobalAveragePooling2D()(base)
    #
    #
    #     excitation = Dense(units=256)(h12)
    #     excitation = Activation('relu')(excitation)
    #     excitation = Dense(units=1024)(excitation)
    #     excitation = Activation('sigmoid')(excitation)
    #
    #     scale = multiply([excitation, base])
    #
    #     x = concatenate([base, scale])
    #     predictions = Dense(10, activation="softmax", kernel_initializer='he_normal')(x)
    #     # model = Model(inputs=[base_model.input,h1], outputs=predictions)
    #     model = keras.Model(inputs=base_model.input, outputs=predictions)
    #     return model
    def build_model(self):

        # # h0 = keras.Input(shape=(224, 224, 3),name="name='input_a")
        # # h1 = keras.Input(shape=(224, 224, 3),name="name='input_b")
        # #
        # # base_model = tensorflow.keras.applications.MobileNet(include_top=False, weights='imagenet', input_tensor=h0)
        # # # base = base_model.output
        # # base = base_model(h0)
        # #
        # # h1 = Conv2D(32, (3, 3), strides=2, padding="same")(h1)
        # # h2 = BatchNormalization()(h1)
        # # h3 = Activation('relu')(h2)
        # #
        # # h4 = self.ca_block(h3,name=str(uuid.uuid4()))
        # # h3 = tf.keras.layers.concatenate([h4, h3], name=str(uuid.uuid4()))
        # #
        # # h4 = self.depth_point_conv2d(h3, s=[1, 1, 2, 1], channel=[64, 128])
        # # h5 = self.depth_point_conv2d(h4, s=[1, 1, 2, 1], channel=[128, 256])
        # # h6 = self.depth_point_conv2d(h5, s=[1, 1, 2, 1], channel=[256, 512])
        # # h7 = self.repeat_conv(h6)
        # # h8 = self.repeat_conv(h7)
        # # h9 = self.repeat_conv(h8)
        # # h10 = self.repeat_conv(h9)
        # #
        # # # h4 = self.ca_block(h10, name=str(uuid.uuid4()))
        # # # h10 = tf.keras.layers.concatenate([h4, h10], name=str(uuid.uuid4()))
        # #
        # # h11 = self.depth_point_conv2d(h10, s=[1, 1, 2, 1], channel=[512, 1024])
        # # h12 = self.repeat_conv(h11, channel=1024)
        # #
        # # h12 = GlobalAveragePooling2D()(h12)
        # # base = GlobalAveragePooling2D()(base)
        # #
        # # excitation = Dense(units=1024 // 16)(h12)
        # # excitation = Activation('relu')(excitation)
        # # excitation = Dense(units=1024)(excitation)
        # # excitation = Activation('sigmoid')(excitation)
        # #
        # # scale = multiply([excitation, base])
        # # xxx = multiply([base, h12])
        # #
        # # x = concatenate([xxx,base,scale])
        # # predictions = Dense(10, activation="softmax", kernel_initializer='he_normal')(x)
        # # # model = Model(inputs=[base_model.input,h1], outputs=predictions)
        # # model = keras.Model(inputs=[base_model.input,h1], outputs=predictions)
        #
        # inp1 = Input(shape=(128, 128, 1), name="img")  # 第一个输入，图像,名字为'img'
        # inp2 = Input(shape=(224, 224, 3), name="rate")  # 第二个输入，条件，名字为'rate'
        #
        # base_model = tensorflow.keras.applications.MobileNet(include_top=False, weights='imagenet',
        #                                                      input_shape=(224, 224, 3))
        # base = base_model(inp2)
        # base = self.py_block(base, 128, 256)
        #
        # x1 = Conv2D(32, 3, padding='same', activation="relu",
        #             # depthwise_regularizer=regularizers.l2(self.weight_decay),
        #             # pointwise_regularizer=regularizers.l2(self.weight_decay),
        #             kernel_initializer='he_normal', name='conv2d_1_1')(inp1)
        # x = MaxPooling2D()(x1)
        # x = BatchNormalization()(x)
        # x2 = SeparableConv2D(64, 5, padding='same', activation="relu",
        #                      # depthwise_regularizer=regularizers.l2(self.weight_decay),
        #                      # pointwise_regularizer=regularizers.l2(self.weight_decay),
        #                      kernel_initializer='he_normal', name='conv2d_1_2')(x)
        # x = MaxPooling2D()(x2)
        # x = BatchNormalization()(x)
        # x3 = SeparableConv2D(128, 7, padding='same', activation="relu",
        #                      # depthwise_regularizer=regularizers.l2(self.weight_decay),
        #                      # pointwise_regularizer=regularizers.l2(self.weight_decay),
        #                      kernel_initializer='he_normal', name='conv2d_1_3')(x)
        # x = MaxPooling2D()(x3)
        # x = BatchNormalization()(x)
        # x4 = SeparableConv2D(256, 9, padding='same', activation="relu",
        #                      # depthwise_regularizer=regularizers.l2(self.weight_decay),
        #                      # pointwise_regularizer=regularizers.l2(self.weight_decay),
        #                      kernel_initializer='he_normal', name='conv2d_1_4')(x)
        # x = MaxPooling2D()(x4)
        # x = BatchNormalization()(x)
        # x = Dropout(0.3)(x)
        #
        # base = GlobalAveragePooling2D()(base)
        # x = GlobalMaxPooling2D()(x)
        # x = tf.keras.layers.multiply([base, x])
        # # base = GlobalAveragePooling2D()(base)
        #
        # # excitation = Dense(units=256)(base)
        # # excitation = Activation('relu')(excitation)
        # # excitation = Dense(units=1024)(base)
        # # excitation = Activation('sigmoid')(excitation)
        #
        # # x = cbam.cbam_module(x)
        #
        # # base = GlobalAveragePooling2D(name=str(uuid.uuid4()))(base)
        # # print(base.shape[-1])
        # # x2 = Dense(1024,activation="relu")(x)
        # # x3 = GlobalMaxPooling2D(name=str(uuid.uuid4()))(x8)
        #
        # # x = tf.keras.layers.concatenate([x4,x5,x3], name=str(uuid.uuid4()))
        # # x = Dense(128, activation='relu')(x)
        # # excitation = Dense(units=256)(x)
        # # excitation = Activation('relu')(excitation)
        # # excitation = Dense(units=1024)(excitation)
        # # excitation = Activation('sigmoid')(excitation)
        #
        # # x1 = tf.keras.layers.multiply([x2, base], name=str(uuid.uuid4()))
        # #         #
        # #         # x = concatenate([base, x1,x])
        #
        # predictions = Dense(10, activation="softmax", kernel_initializer='he_normal')(x)
        # model = Model(inputs=[inp1, inp2], outputs=predictions)  # inp1和inp2作为输入，输出为out的模型实例化
        # model.summary()
        # return model
        # base_model = tensorflow.keras.applications.EfficientNetB0(include_top=False, weights="imagenet")
        inputs = tensorflow.keras.Input(shape=(224, 224, 3), name="name='input_a")

        base_model = tensorflow.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
        # x = Conv2D(64,7,strides=2)(base_model.get_layer('block1a_activation').output)
        # x = self.SKBlock(x,64)
        # print(base_model.summary())
        x = mobilenet_block(base_model.get_layer('block1a_activation').output, filters=64, kernel_size=7, strides=1)
        # x = mobilenet_block(x, filters=64, kernel_size=7, strides=1)

        # x = mobilenet_block(x, filters=128, kernel_size=7, strides=2)
        # x = mobilenet_block(x, filters=128, kernel_size=7, strides=1)


        # base_model = keras_cv_attention_models.mobilevit.MobileViT(input_shape=(224,224,3),pretrained='imagenet')
        # # base = base_model.output
        # x = base_model.layers[-2].output
        # x = mobilenet_block(base_model.get_layer('stack2_block1_deep_1_swish').output, filters=32, kernel_size=3, strides=2)
        # # base = base_model(inputs)

        # base_model.summary()
        # base1 = GlobalAveragePooling2D()(base)
        # 1) MobileNet 前段
        # x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(base_model.get_layer('block1a_activation').output)
        # x = layers.BatchNormalization()(x)
        # x = layers.ReLU(max_value=6)(x)
        # x = mobilenet_block(base_model.get_layer('block1a_activation').output, 64, strides=1)  # 112×112×64
        # MobileNet Block
        # # base = base_model(inputs)

        # base_model.summary()
        # base1 = GlobalAveragePooling2D()(base)
        # 1) MobileNet 前段
        # x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(base_model.get_layer('block1a_activation').output)
        # x = layers.BatchNormalization()(x)
        # x = layers.ReLU(max_value=6)(x)
        # x = mobilenet_block(base_model.get_layer('block1a_activation').output, 64, strides=1)  # 112×112×64
        # MobileNet Block

        # EfficientNet Block
        x = efficientnet_block(x, filters=128, kernel_size=3, strides=1, expand_ratio=6)
        x = efficientnet_block(x, filters=128, kernel_size=3, strides=2, expand_ratio=6)
        # # 空间注意力转换器
        # _, h, w, c = x.shape
        # x_reshape = layers.Reshape((h * w, c))(x)  # (batch, seq_len, embed_dim)
        #
        # # 轻量级Transformer
        # transformer_block = TransformerBlock(embed_dim=c, num_heads=4, ff_dim=128)
        # x_trans = transformer_block(x_reshape)
        # x_trans = layers.Reshape((h, w, c))(x_trans)
        #
        # # 合并原始特征和注意力特征
        # x = layers.Concatenate()([x, x_trans])

        # ConvNeXt Block
        # x = convnext_block(x, filters=128)
        # x = convnext_block(x, filters=128)
        x = convnext_block(x, filters=256)
        x = convnext_block(x, filters=256)
        x = self.ca_block(x)
        x = self.SKBlock(x, 256)
        # x = cbam.cbam_module(x)

        x = layers.Conv2D(base_model.output.shape[-1], kernel_size=1, padding='same')(x)
        # 分类头
        x = layers.GlobalAveragePooling2D()(x)
        print(base_model.output)
        base = GlobalAveragePooling2D()(base_model.output)
        scale = multiply([x, base])
        x1 = self.dynamic_weight_fusion(base,scale)
        # x = concatenate([base,scale])
        # predictions1 = Dense(10, activation="softmax", kernel_initializer='he_normal')(x)
        # predictions2 = Dense(10, activation="softmax", kernel_initializer='he_normal')(base1)
        predictions = Dense(10, activation="softmax", kernel_initializer='he_normal')(x1)

        # model = Model(inputs=[base_model.input,h1], outputs=predictions)
        model = keras.Model(inputs=inputs, outputs=predictions)
        model.summary()
        return model



    def SKBlock(self,input_tensor, filters):
        # 分支1: 3x3卷积
        # branch1 = SeparableConv2D(filters, 3, padding='same', activation='relu')(input_tensor)

        branch2 = SeparableConv2D(filters, 5, padding='same', activation='relu')(input_tensor)
        # 分支2: 5x5卷积（等效两个3x3）
        branch3 = SeparableConv2D(filters, 7, padding='same', activation='relu')(input_tensor)

        # 注意力融合
        merged = Concatenate()([branch2,branch3])
        gap = GlobalAvgPool2D()(merged)
        fc = Dense(filters//8, activation='relu')(gap)
        attn = Dense(filters * 2, activation='softmax')(fc)  # 动态权重
        attn_b2, attn_b3 = tf.split(attn, num_or_size_splits=2, axis=1)
        output = Multiply()([branch2, attn_b2])+Multiply()([branch3, attn_b3])
        # 残差连接
        shortcut = Conv2D(filters, 1)(input_tensor) if input_tensor.shape[-1] != filters else input_tensor
        return tf.keras.layers.Add()([output, shortcut])
    def dynamic_weight_fusion(self, base, scale):
        # ‌通道注意力增强版（CA - EfficientNet改进方案）‌
        # 动态权重生成（双层Dense）
        weights = layers.Concatenate()([base, scale])
        # weights = BatchNormalization()(weights)
        weights = layers.Dense(64, activation='relu')(weights)
        weights = layers.Dense(2, activation='softmax')(weights)
        # 加权融合
        weighted_base = layers.Multiply()([base, weights[:, 0:1]])
        weighted_scale = layers.Multiply()([scale, weights[:, 1:2]])
        return layers.Add()([weighted_base, weighted_scale])

    def train(self):
        X_train = np.load('./driver_feature_small_driver100/train/images.npy')
        X_train = X_train.astype(np.float16)
        X_train = X_train.reshape([-1, 224, 224, 3])
        np.random.seed(42)
        np.random.shuffle(X_train)

        y_train = np.load('./driver_feature_small_driver100/train/labels.npy')
        np.random.seed(42)
        np.random.shuffle(y_train)

        # X_val = np.load('./driver_feature_small_driver100/val/images.npy')
        # X_val = X_val.astype(np.float16)
        # X_val = X_val.reshape([-1, 224, 224, 3])
        # np.random.seed(42)
        # np.random.shuffle(X_val)
        #
        # y_val = np.load('./driver_feature_small_driver100/val/labels.npy')
        # np.random.seed(42)
        # np.random.shuffle(y_val)

        X_test = np.load('./driver_feature_small_driver100/test/images.npy')
        X_test = X_test.astype(np.float16)
        X_test = X_test.reshape([-1, 224, 224, 3])
        np.random.seed(42)
        np.random.shuffle(X_test)

        y_test = np.load('./driver_feature_small_driver100/test/labels.npy')
        np.random.seed(42)
        np.random.shuffle(y_test)

        print(X_train.shape)
        print(y_train.shape)

        # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=2025)
        print(X_train.shape)
        print(y_train.shape)
        print(X_test.shape)
        print(y_test.shape)

        # 创建Momentum优化器
        momentum_optimizer = tf.keras.optimizers.SGD(learning_rate=0.0015, momentum=0.95)
        self.model.compile(optimizer=momentum_optimizer,
                           loss='categorical_crossentropy', # 损失函数
                              metrics=['accuracy'])  # 指标
        # training the model
        metrics = Metrics()
        self.model.validation_data = (X_test, y_test)
        # 创建Momentum优化器
        # 定义学习率衰减回调
        reduce_lr = ReduceLROnPlateau(
            monitor='val_accuracy',  # 监控验证损失
            factor=0.2,  # 学习率下降因子
            mode="max",
            verbose=1,
            patience=3,  # 3轮未改善则触发
            min_lr=1e-6  # 最小学习率
        )
        history = self.model.fit(X_train, y_train,
                                 batch_size=16,
                                 epochs=self.epoch,
                                 verbose=1,
                                 shuffle=True,
                                 validation_data=(X_test, y_test),
                                 callbacks=[metrics,reduce_lr]
                                 )

        # yPred = []
        # y = []
        # for i in range(len(X_valid)):
        #     image = np.expand_dims(X_valid[i], axis=0)
        #     predictions = self.model.predict(image)
        #     top_class_index = tf.argmax(predictions, axis=-1)
        #     # top_class_probability = predictions[0][top_class_index]
        #     yPred.append([int(top_class_index.numpy())])
        #     x_real = [k for k, v in enumerate(y_valid[i]) if v == 1]
        #     y.append(x_real)
        # draw_confu(y, yPred, name='train')
        # # self.model.save('EffNASNet.h5')
        #
        # _val_f1 = f1_score(y, yPred, average='micro')
        # _val_recall = recall_score(y, yPred, average='micro')
        #
        # print("draw compliment" + str(_val_f1) + "*---" + str(_val_recall))



        # 靠谱方式
        data_path_abs = './auc/auc'
        img_list_all = os.listdir(data_path_abs)
        for key, v in enumerate(img_list_all):
            input_img1 = cv2.imread(data_path_abs + "/" + v)
            input_img1 = cv2.resize(input_img1, (224, 224))
            input_img1 = np.expand_dims(input_img1, axis=0)
            # sobel_image = sobel_image - np.mean(sobel_image, axis=0)
            predictions = self.model.predict(input_img1)
            #     # 获取最后一层卷积层的输出
            last_conv_layer = self.model.get_layer('top_conv')
            grad_model = Model([self.model.inputs], [last_conv_layer.output, self.model.output])
            #     # 计算类别的梯度
            with tf.GradientTape() as tape:
                conv_layer_output, preds = grad_model(input_img1)
                class_channel = preds[0][np.argmax(preds[0])]
            # 计算梯度
            grads = tape.gradient(class_channel, conv_layer_output)
            # 计算权重
            print(v, end="------------")
            print(preds[0], end="------------np.argmax(preds[0]):")
            print(np.argmax(preds[0]), end="------------class_channel:")
            # class_channel = preds[0][np.argmax(preds[0])]
            print(class_channel)
            print("==================================================")

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            # pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
            # heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_layer_output), axis=-1)
            heatmap = conv_layer_output @ pooled_grads[..., tf.newaxis]
            # feature_map_sum = sum(ele for ele in feature_map_combination)
            heatmap = tf.maximum(heatmap, 0)
            heatmap /= tf.reduce_max(heatmap)

            # 重塑热力图并将其缩放到与原始图像相同的大小
            heatmap = np.squeeze(heatmap)
            # 颜色映射
            gbkInput = cv2.imread(data_path_abs + "/" + v)
            gbkInput = cv2.resize(gbkInput, (224, 224))

            feature_map_sum1 = cv2.resize(heatmap, (224, 224))
            # 将热力图转换为RGB格式
            feature_map_sum = np.uint8(255 * feature_map_sum1)
            feature_map_sum[feature_map_sum < 80] = 0
            # 将热利用应用于原始图像
            feature_map_sum = cv2.applyColorMap(feature_map_sum, cv2.COLORMAP_JET)
            # 　这里的热力图因子是０.４
            superimposed_img = feature_map_sum * 0.4 + gbkInput
            cv2.imwrite( "person{}".format(v+".jpg"), superimposed_img)
            # cv2.imwrite( "person{}".format(v+".eps"), superimposed_img)
            plt.savefig("{}".format("person" + str(uuid.uuid4()) + ".eps"))

            print("热力图完成")
        #
        #     #
        #     # gbkInput = cv2.imread(data_path_abs + "/" + v)
        #     # gbkInput = cv2.resize(gbkInput, (224, 224))
        #     # last_conv_layer = self.model.get_layer('conv2d_1_4')  # 根据你的模型调整层名
        #     # last_conv_layer_output = last_conv_layer.output
        #     # model_with_last_conv = Model(inputs=self.model.input, outputs=last_conv_layer_output)
        #     # feature_map = model_with_last_conv.predict(sobel_image)
        #     # feature_map_sum = visualize_feature_map(feature_map)
        #     # #
        #     # # 将热力图的大小调整与原图一致
        #     # feature_map_sum = cv2.resize(feature_map_sum, (gbkInput.shape[1], gbkInput.shape[0]))
        #     # # 将热力图转换为RGB格式
        #     # feature_map_sum = np.uint8(255 * feature_map_sum)
        #     # print(feature_map_sum.shape)
        #     # 将热利用应用于原始图像
        #     # feature_map_sum = cv2.applyColorMap(feature_map_sum, cv2.COLORMAP_JET)
        #     # # 　这里的热力图因子是０.４
        #     #
        #     # superimposed_img = feature_map_sum * 0.4 + gbkInput
        #     # print(superimposed_img.shape)
        #     # # cv2.imshow('Segmentation', superimposed_img)
        #     # # cv2.waitKey(1500)
        #     # # cv2.destroyAllWindows()
        #     # # gbkInput = cv2.resize(gbkInput, (28, 28))
        #     # # gbkInput = cv2.cvtColor(gbkInput, cv2.COLOR_BGR2GRAY)
        #     # # print(gbkInput.shape)
        #     # # super_imposed_img1 = feature_map_sum * 0.3 + gbkInput
        #     #
        #     # cv2.imwrite( "person{}".format(v+".jpg"), superimposed_img)
        #     #
        #     # # plt.savefig("{}".format("feature_map_sum" + str(uuid.uuid4()) + ".eps"))
        #     # # plt.savefig("{}".format("feature_map_sum" + str(uuid.uuid4()) + ".jpg"))
        #
        # yPred = []
        # y = []
        #
        # conficent = []
        # temp = []
        #
        # for i in range(len(X_valid)):
        #     image = np.expand_dims(X_valid[i], axis=0)
        #     predictions = self.model.predict(image)
        #     top_class_index = tf.argmax(predictions, axis=-1)
        #     top_class_probability = predictions[0][top_class_index]
        #     x_real = [k for k, v in enumerate(y_valid[i]) if v == 1]
        #     real_class_probability = predictions[0][x_real[0]]
        #     yPred.append([int(top_class_index.numpy())])
        #     temp.append(int(top_class_index.numpy()))
        #     temp.append(top_class_probability)
        #     temp.append(x_real[0])
        #     temp.append(real_class_probability)
        #     conficent.append(temp)
        #     temp = []
        #     y.append(x_real)
        #
        # with open("confidenceFile/auc.csv", "a", newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["top_class_index", "top_class_probability", "x_real", "real_class_probability"])
        #     for r in conficent:
        #         writer.writerow(r)
        # draw_confu(y, yPred, name='train')
        #
        #
        #
        # fer_json = self.model.to_json()
        # with open("xiaorong_cam.json", "w") as json_file:
        #     json_file.write(fer_json)
        # self.model.save_weights("xiaorong_cam.h5")
        # print("Saved model to disk")

def draw_confu(y, y_pred, name=''):
    sns.set(font_scale=3)
    confusion_matrix = sklearn.metrics.confusion_matrix(y, y_pred)
    plt.xticks(fontsize=10)  # 设置x轴刻度字体大小为12
    plt.yticks(fontsize=10)
    plt.figure(figsize=(16, 14))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20});
    plt.title("Confusion Matrix", fontsize=32)
    plt.ylabel('Actual Label', fontsize=28)
    plt.xlabel('Predicted Label', fontsize=28)
    plt.savefig('./result_%s.eps' % (name))
    plt.savefig('./result_%s.svg' % (name))
    plt.savefig('./result_%s.jpg' % (name))

def evaluate(model, X, Y):
    accuracy = model.evaluate(X, Y)
    return accuracy[0]

def get_flops(model, model_inputs) -> float:
    """
    Calculate FLOPS [GFLOPs] for a tf.keras.Model or tf.keras.Sequential model
    in inference mode. It uses tf.compat.v1.profiler under the hood.
    """
    # if not hasattr(model, "model"):
    #     raise wandb.Error("self.model must be set before using this method.")

    if not isinstance(
            model, (tf.keras.models.Sequential, tf.keras.models.Model)
    ):
        raise ValueError(
            "Calculating FLOPS is only supported for "
            "`tf.keras.Model` and `tf.keras.Sequential` instances."
        )

    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph,
    )

    # Compute FLOPs for one sample
    batch_size = 1
    inputs = [
        tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype)
        for inp in model_inputs
    ]

    # convert tf.keras model into frozen graph to count FLOPs about operations used at inference
    real_model = tf.function(model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

    # Calculate FLOPs with tf.profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = (
        tf.compat.v1.profiler.ProfileOptionBuilder(
            tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
        )
            .with_empty_output()
            .build()
    )

    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )

    tf.compat.v1.reset_default_graph()

    # convert to GFLOPs
    return (flops.total_float_ops / 1e9) / 2

if __name__ == '__main__':
    seed_value = 2025
    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    import os
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['HOROVOD_FUSION_THRESHOLD'] = '0'
    import random

    random.seed(seed_value)
    import numpy as np

    np.random.seed(seed_value)
    import tensorflow as tf

    tf.compat.v1.set_random_seed(seed_value)
    from keras import backend as K

    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    K.set_session(sess)

    # 不使用gpu则开启这一行代码
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(tf.test.is_gpu_available())
    fer_model = FerModel()
    print(tf.test.is_gpu_available())


