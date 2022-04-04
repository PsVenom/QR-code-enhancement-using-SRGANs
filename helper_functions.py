@tf.function
def build_data(data):
  cropped=tf.dtypes.cast(tf.image.random_crop(data['image'] / 255,(128,128,3)),tf.float32)
  lr=tf.image.resize(cropped,(32,32))
  return (lr,cropped * 2 - 1)
train_dataset_mapped = train_data.map(build_data,num_parallel_calls=tf.data.AUTOTUNE)
for x in train_dataset_mapped.take(1):
  plt.imshow(x[0].numpy())
 plt.show()
  plt.imshow(bicubic_interpolate(x[0].numpy(),(128,128)))
  plt.show()
  plt.imshow(x[1].numpy())
  plt.show()
  
  
  
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
input_lr=tf.keras.layers.Input(shape=(None,None,3))
input_conv=tf.keras.layers.Conv2D(64,9,padding='same')(input_lr)
input_conv=tf.keras.layers.LeakyReLU()(input_conv)
SRRes=input_conv
for x in range(5):
  res_output=residual_block_gen()(SRRes)
  SRRes=tf.keras.layers.Add()([SRRes,res_output])
SRRes=tf.keras.layers.Conv2D(64,9,padding='same')(SRRes)
SRRes=tf.keras.layers.BatchNormalization()(SRRes)
SRRes=tf.keras.layers.Add()([SRRes,input_conv])
SRRes=Upsample_block(SRRes)
SRRes=Upsample_block(SRRes)
output_sr=tf.keras.layers.Conv2D(3,9,activation='tanh',padding='same')(SRRes)
SRResnet=tf.keras.models.Model(input_lr,output_sr)
