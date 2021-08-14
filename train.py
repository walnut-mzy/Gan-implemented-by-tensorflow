from model import Generator,Discriminator
from tensorflow import keras
import os
import settings
import tensorflow as tf
from utils1 import d_loss_fn,g_loss_fn,load,save_result,dataset1

generator = Generator() # 创建生成器
generator.build(input_shape = (settings.batch_size, settings.z_dim))
discriminator = Discriminator() # 创建判别器
discriminator.build(input_shape=(settings.batch_size, settings.image_size, settings.image_size, settings.chanl))
g_optimizer = keras.optimizers.Adam(learning_rate=settings.learning_rate, beta_1=0.5)
d_optimizer = keras.optimizers.Adam(learning_rate=settings.learning_rate, beta_1=0.5)

@tf.function
def train_step(batch_x):
    # 采样隐藏向量
    batch_z = tf.random.normal([settings.batch_size, settings.z_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # 采样生成图片
        fake_image = generator(batch_z, training=True)
        # 判定生成图片
        d_fake_logits = discriminator(fake_image, training=True)
        # 判定真实图片
        d_real_logits = discriminator(batch_x, training=True)
        d_loss = d_loss_fn(d_real_logits, d_fake_logits)
        g_loss = g_loss_fn(d_fake_logits)
    grads_d = disc_tape.gradient(d_loss, discriminator.trainable_variables)
    grads_g = gen_tape.gradient(g_loss, generator.trainable_variables)
    d_optimizer.apply_gradients(zip(grads_d, discriminator.trainable_variables))
    g_optimizer.apply_gradients(zip(grads_g, generator.trainable_variables))

    return d_loss, g_loss


if __name__ == '__main__':
    if not os.path.exists(settings.OUTPUT_DIR):
        os.mkdir(settings.OUTPUT_DIR)
    for epoch in range(settings.epochs):
        for n, data in dataset1.enumerate():
            d_loss, g_loss = train_step(data)
            print('.', end='')
            if n % 100 == 0:
                print(n.numpy(), 'd-loss:', float(d_loss), 'g-loss:', float(g_loss))
                # 可视化
                z = tf.random.normal([100, settings.z_dim])
                fake_image = generator(z, training=False)
                img_path = os.path.join(settings.OUTPUT_DIR, 'gan-%d.png' % n)
                save_result(fake_image.numpy(), 10, img_path, color_mode='P')
        try:
            generator.save("generator.h5")
            discriminator.save("discriminator.h5")
        except:
            print("error")