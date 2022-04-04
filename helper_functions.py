import tensorflow as tf
import matplotlib.pyplot as ple
@tf.function
def build_data(data):
  cropped=tf.dtypes.cast(tf.image.random_crop(data['image'] / 255,(128,128,3)),tf.float32)
  lr=tf.image.resize(cropped,(32,32))
  return (lr,cropped * 2 - 1)

@tf.function()
def train_step(data,loss_func=pixel_MSE,adv_learning=True,evaluate=['PSNR'],adv_ratio=0.001):
  logs={}
  gen_loss,disc_loss=0,0
  low_resolution,high_resolution=data
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    super_resolution = SRResnet(low_resolution, training=True)
    gen_loss=loss_func(high_resolution,super_resolution)
    logs['reconstruction']=gen_loss
    if adv_learning:
      real_output = discriminator(high_resolution, training=True)
      fake_output = discriminator(super_resolution, training=True)
      
      adv_loss_g = generator_loss(fake_output) * adv_ratio
      gen_loss += adv_loss_g
      
      disc_loss = discriminator_loss(real_output, fake_output)
      logs['adv_g']=adv_loss_g
      logs['adv_d']=disc_loss
  gradients_of_generator = gen_tape.gradient(gen_loss, SRResnet.trainable_variables)
  generator_optimizer.apply_gradients(zip(gradients_of_generator, SRResnet.trainable_variables))
  
  if adv_learning:
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
  
  for x in evaluate:
    if x=='PSNR':
      logs[x]=PSNR(high_resolution,super_resolution)
  return logs
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
cross_entropy = tf.keras.losses.BinaryCrossentropy()
def discriminator_loss(real_output, fake_output):
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
  total_loss = real_loss + fake_loss
  return total_loss
def generator_loss(fake_output):
  return cross_entropy(tf.ones_like(fake_output), fake_output)
