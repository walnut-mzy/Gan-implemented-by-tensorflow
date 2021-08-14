import tensorflow as tf
from tensorflow import keras
import settings
from PIL import Image
import os
import numpy as np
def celoss_ones(logits):
    # 计算属于与标签为 1 的交叉熵
    y = tf.ones_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)
def celoss_zeros(logits):
    # 计算属于与便签为 0 的交叉熵
    y = tf.zeros_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def g_loss_fn(d_fake_logits):
    # 计算生成图片与1之间的误差
    loss = tf.reduce_mean(loss_object(tf.ones_like(d_fake_logits), d_fake_logits))

    return loss


def d_loss_fn(d_real_logits, d_fake_logits):
    # 真实图片与1之间的误差
    d_loss_real = tf.reduce_mean(loss_object(tf.ones_like(d_real_logits), d_real_logits))
    # 生成图片与0之间的误差
    d_loss_fake = tf.reduce_mean(loss_object(tf.zeros_like(d_fake_logits), d_fake_logits))
    # 合并误差
    loss = d_loss_fake + d_loss_real

    return loss

def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    #image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [settings.image_size, settings.image_size])
    image = (image - 127.5) / 127.5
    print(image)
    return image
def train_test_get(train_test_inf):
    for root,dir,files in os.walk(train_test_inf, topdown=False):
        #print(root)
        #print(files)
        list1=[root+"/"+i for i in files]
        return list1
dataset=tf.data.Dataset.list_files("./ktFaces/*.jpg")
dataset1 = dataset.map(
    load, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(
    settings.SHUFFLE_SIZE).batch(settings.batch_size).repeat(100)
#.cache().shuffle(settings.SHUFFLE_SIZE).batch(settings.batch_size).repeat(100)
def save_result(val_out, val_block_size, image_path, color_mode):
    preprocesed = ((val_out + 1.0) * 127.5).astype(np.uint8)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b+1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])
    Image.fromarray(final_image).save(image_path)
