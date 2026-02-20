import csv

from keras_cv_attention_models import cmt
from keras_cv_attention_models.efficientformer import EfficientFormerV2
from keras_cv_attention_models.efficientvit import EfficientViT_B0
from tensorflow.keras import backend as K
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
    maxAcc=0
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
            if val_accuracy is not None and val_accuracy>=0.9566 and val_accuracy>Metrics.maxAcc:
                # print(f"Epoch {epoch + 1}: Validation Accuracy = {val_accuracy:.4f}")
                fer_json = self.model.to_json()
                with open("bestParam.json", "w") as json_file:
                    json_file.write(fer_json)
                self.model.save_weights("bestParam.h5")
                Metrics.maxAcc = val_accuracy
                print(Metrics.maxAcc)
                print("Saved model to disk")

        _val_recall = recall_score(val_targ, val_predict,average='macro')
        _val_precision = precision_score(val_targ, val_predict,average='macro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print (" — val_f1: % f — val_precision: % f — val_recall % f" % (_val_f1, _val_precision, _val_recall))
        return
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class MemoryMonitor(Callback):
    def on_epoch_end(self, epoch, logs=None):
        mem_info = tf.config.experimental.get_memory_info('GPU:0')
        print(f"Epoch {epoch+1} - GPU显存占用: {mem_info['current']/1e9:.2f}GB")



import uuid
# 用于避免卷积层同名报错
unique_random_number = uuid.uuid4()
from tensorflow import keras
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
        # self.history = self.train()
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

    def build_model(self):
        inputs = tensorflow.keras.Input(shape=(224, 224, 3), name="name='input_a")
        base_model = tensorflow.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
        base = GlobalAveragePooling2D()(base_model.output)
        predictions = Dense(10, activation="softmax", kernel_initializer='he_normal')(base)
        model = keras.Model(inputs=inputs, outputs=predictions)
        # model.summary()
        return model

        # from keras_cv_attention_models import swin_transformer_v2, repvit, mobilevit, convnext, uniformer, iformer, \
        #     caformer, davit, \
        #     maxvit, fastvit, flexivit, efficientformer, gpvit, gcvit, tinyvit, pvt, levit, coat, edgenext, resnext, \
        #     lcnet, efficientnet, halonet, efficientdet, fbnetv3, meta_transformer, nfnets, maxvit
        # model = swin_transformer_v2.SwinTransformerV2(input_shape=(224, 224, 3), pretrained=None)
        # model = repvit.RepViT(input_shape=(224, 224, 3),pretrained=None)
        # model = mobilevit.MobileViT(input_shape=(224, 224, 3),pretrained="imagenet")
        # model = uniformer.Uniformer(input_shape=(224, 224, 3),pretrained=None)
        # model = EfficientFormerV2(input_shape=(224, 224, 3),pretrained=None)
        # model = EfficientViT_B0(input_shape=(224, 224, 3),pretrained='None')
        # model = efficientformer.EfficientFormer(input_shape=(224, 224, 3),pretrained=Nonde)
        # model = gcvit.GCViT(input_shape=(224, 224, 3),pretrained=None)
        # model = caformer.ConvFormerS18(input_shape=(224, 224, 3),pretrained=None)
        # model = fastvit.FastViT(input_shape=(224, 224, 3),pretrained=None)
        # model = flexivit.FlexiViT(input_shape=(224, 224, 3),pretrained=None)
        # model = davit.DaViT(input_shape=(224, 224, 3),pretrained=None)
        # model = tinyvit.TinyViT(input_shape=(224, 224, 3),pretrained=None)
        # model = cmt.CMTTiny (input_shape=(224, 224, 3),pretrained=None)
        # model = lcnet.LCNet(input_shape=(224, 224, 3),pretrained=None)
        # model = EfficientFormerV2(input_shape=(224, 224, 3),pretrained=None)
        # model = efficientnet.EfficientNetV2(input_shape=(224, 224, 3),pretrained=None)
        # model = efficientnet.EfficientNetV1(input_shape=(224, 224, 3),pretrained=None)
        # model = mobilevit.MobileViT_V2(input_shape=(224, 224, 3),pretrained=None)
        # model = resnext.ResNeXt50(input_shape=(224, 224, 3),pretrained=None)
        # model = fbnetv3.FBNetV3(input_shape=(224, 224, 3),pretrained=None)
        # model = fbnetv3.FBNetV3D(input_shape=(224, 224, 3),pretrained=None)
        # model = fbnetv3.FBNetV3G(input_shape=(224, 224, 3),pretrained=None)
        # model = fbnetv3.FBNetV3B(input_shape=(224, 224, 3),pretrained=None)
        # model = pvt.PyramidVisionTransformerV2(input_shape=(224, 224, 3),pretrained="imagenet")
        # model = mobilevit.MobileViT_S(input_shape=(224, 224, 3),pretrained=None)
        # model = efficientformer.EfficientFormerL1(input_shape=(224, 224, 3),pretrained=None)
        # model = efficientformer.EfficientFormerV2L(input_shape=(224, 224, 3),pretrained=None)
        #
        # x = model.layers[-2].output
        # outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
        # custom_model = tf.keras.Model(model.input, outputs)
        # # custom_model.summary()
        # return custom_model


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
        import numpy as np
        X_train = np.load('./driver_feature_auc_four/train/images1.npy')
        X_train = X_train.reshape([-1, 224, 224, 3])
        np.random.seed(2025)
        np.random.shuffle(X_train)

        y_train = np.load('./driver_feature_auc_four/train/labels.npy')
        np.random.seed(2025)
        np.random.shuffle(y_train)

        X_valid = np.load('./driver_feature_auc_four/test/images1.npy')
        X_valid = X_valid.reshape([-1, 224, 224,3])
        np.random.seed(2025)
        np.random.shuffle(X_valid)

        y_valid = np.load('./driver_feature_auc_four/test/labels.npy')
        np.random.seed(2025)
        np.random.shuffle(y_valid)

        print(X_train.shape)
        print(y_train.shape)
        print(X_valid.shape)
        print(y_valid.shape)

        # 创建Momentum优化器
        momentum_optimizer = tf.keras.optimizers.SGD(learning_rate=0.0015, momentum=0.95)
        self.model.compile(optimizer=momentum_optimizer,
                           loss='categorical_crossentropy', # 损失函数
                              metrics=['accuracy'])  # 指标
        # training the model
        metrics = Metrics()
        self.model.validation_data = (X_valid, y_valid)
        # 创建Momentum优化器

        from tensorflow.keras.callbacks import ReduceLROnPlateau
        #
        # reduce_lr = ReduceLROnPlateau(
        #     monitor='val_loss',  # 推荐：监控验证损失
        #     factor=0.2,
        #     patience=5,
        #     verbose=1,
        #     mode='min',
        #     min_delta=0.0001,
        #     cooldown=2,
        #     min_lr=1e-6
        # )

        # 定义学习率衰减回调
        reduce_lr = ReduceLROnPlateau(

            monitor='val_accuracy',  # 监控验证损失
            factor=0.2,  # 学习率下降因子
            mode="max",
            verbose=1,
            patience=5,  # 3轮未改善则触发
            min_lr=1e-6  # 最小学习率
        )
        history = self.model.fit(X_train, y_train,
                                 batch_size=16,
                                 epochs=self.epoch,
                                 verbose=1,
                                 shuffle=True,
                                 validation_data=(X_valid, y_valid),
                                 callbacks=[metrics,reduce_lr]
                                 )

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
    tf.random.set_seed(2025)

    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    K.set_session(sess)

    # 不使用gpu则开启这一行代码
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(tf.test.is_gpu_available())
    fer_model = FerModel()
    print(tf.test.is_gpu_available())