import tensorflow as tf
import matplotlib.pyplot as ple
@tf.function
def build_data(data):
  cropped=tf.dtypes.cast(tf.image.random_crop(data['image'] / 255,(128,128,3)),tf.float32)
  lr=tf.image.resize(cropped,(32,32))
  return (lr,cropped * 2 - 1)
  
  #defining the resnet
def residual_block_gen(ch=64,k_s=3,st=1):
  model=tf.keras.Sequential([
  tf.keras.layers.Conv2D(ch,k_s,strides=(st,st),padding='same'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.LeakyReLU(),
  tf.keras.layers.Conv2D(ch,k_s,strides=(st,st),padding='same'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.LeakyReLU(),
  ])
  return model
def Upsample_block(x, ch=256, k_s=3, st=1):
  x = tf.keras.layers.Conv2D(ch,k_s, strides=(st,st),padding='same')(x)
  x = tf.nn.depth_to_space(x, 2) # Subpixel pixelshuffler
  x = tf.keras.layers.LeakyReLU()(x)
  return x

def residual_block_disc(ch=64,k_s=3,st=1):
  model=tf.keras.Sequential([
  tf.keras.layers.Conv2D(ch,k_s,strides=(st,st),padding='same'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.LeakyReLU(),
  ])
  return model
