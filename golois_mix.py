import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers 
from tensorflow.keras import regularizers
import gc

import golois

planes = 31
moves = 361
N = 10000
epochs = 20
batch = 128
filters = 64


def SE_Block(t,filters, ratio = 16):
  se_shape = (1, 1, filters)
  se = layers.GlobalAveragePooling2D()(t)
  se = layers.Reshape(se_shape)(se)
  se = layers.Dense(filters // ratio, activation = 'relu', use_bias = False)(se)
  se = layers.Dense(filters, activation = 'sigmoid', use_bias = False)(se)
  x = layers.multiply([t, se])
  return x

def mixconv(x, filters, **args):
  G = len(filters)
  y = []
  for i, (xi, fi) in enumerate(zip(tf.split(x, G, axis=-1), filters)):
    gi = layers.DepthwiseConv2D(2*(i+1) + 1, strides=1,padding= 'same', **args)(xi)
    gi = layers.BatchNormalization()(gi)
    gi = layers.Activation('swish')(gi)
    y.append(gi)
  return tf.concat(y, axis=-1)

def depthwiseconv(x, strides: int):
  x = layers.DepthwiseConv2D(3, strides=strides, padding= 'same' if strides == 1 else 'valid', use_bias= False)(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation('swish')(x)
  return x

def pointwiseconv(x, filters: int, linear: bool):
  x = layers.Conv2D(filters = filters, kernel_size= (1,1), strides= (1,1), padding= 'same', use_bias= False)(x)
  x = layers.BatchNormalization()(x)
  if linear == False:
    x = layers.Activation('swish')(x)
  return x

def standardconv(x):
  x = layers.Conv2D(filters= 32, kernel_size= (1,1), use_bias= False)(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation('swish')(x)
  return x

def inverted_residual_block(x, strides_depthwise: int, filter_pointwise: int, expansion: int):
  x1 = pointwiseconv(x, filters = filter_pointwise * expansion, linear = False)
  #x1 = depthwiseconv(x1, strides_depthwise)
  x1 = mixconv(x1, [filter_pointwise/4 for i in range(4)])
  x1 = pointwiseconv(x1, filters = filter_pointwise, linear = True)
  if strides_depthwise == 1 and x.shape[-1] == filter_pointwise:
    x1 = SE_Block(x1, filter_pointwise)
    return layers.add([x1,x])
  else:
    return x1

def bottleneck_block(x, s: int, c: int, t: int, n: int):
  ''' 
    s : strides
    c : output channels
    t : expansion factor
    n : repeat
  '''
  x = inverted_residual_block(x, strides_depthwise= s, filter_pointwise= c, expansion= t)
  for i in range(n-1):
    x = inverted_residual_block(x, strides_depthwise= 1, filter_pointwise= c, expansion= t)
  return x


input_data = np.random.randint(2, size=(N, 19, 19, planes))
input_data = input_data.astype ('float32')

policy = np.random.randint(moves, size=(N,))
policy = keras.utils.to_categorical (policy)

value = np.random.randint(2, size=(N,))
value = value.astype ('float32')

end = np.random.randint(2, size=(N, 19, 19, 2))
end = end.astype ('float32')

groups = np.zeros((N, 19, 19, 1))
groups = groups.astype ('float32')

print ("getValidation", flush = True)
golois.getValidation (input_data, policy, value, end)


input = keras.Input(shape=(19, 19, planes), name='board')
x = standardconv(input)
x = bottleneck_block(x, s=1, c=16, t=1, n=1)
x = bottleneck_block(x, s=1, c=24, t=2, n=2)
x = bottleneck_block(x, s=1, c=32, t=2, n=3)
x = bottleneck_block(x, s=1, c=64, t=2, n=4)
x = bottleneck_block(x, s=1, c=96, t=1, n=3)
x = bottleneck_block(x, s=1, c=160, t=1, n=3)
x = bottleneck_block(x, s=1, c=320, t=1, n=1)
x = pointwiseconv(x, filters = 1280, linear = False)

'''
for i in range (2):
    x1 = layers.Conv2D(filters, 3, activation='swish', padding='same')(x)
    x1 = layers.Conv2D(filters, 3, padding='same')(x1)
    x1 = layers.GlobalAveragePooling2D()(x1)
    x1 = layers.Dense(filters)(x1)
    x = layers.add([x1,x])
    x = layers.Activation('swish')(x)
    x = layers.BatchNormalization()(x)
'''

policy_head = layers.Conv2D(1, 1, activation='swish', padding='same', use_bias = False, kernel_regularizer=regularizers.l2(0.0001))(x)
policy_head = layers.Flatten()(policy_head)
policy_head = layers.Activation('softmax', name='policy')(policy_head)
value_head = layers.GlobalAveragePooling2D()(x)
value_head = layers.Dense(50, activation='swish', kernel_regularizer=regularizers.l2(0.0001))(value_head)
value_head = layers.Dense(1, activation='sigmoid', name='value', kernel_regularizer=regularizers.l2(0.0001))(value_head)

model = keras.Model(inputs=input, outputs=[policy_head, value_head])

model.summary ()

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
              loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
              loss_weights={'policy' : 1.0, 'value' : 1.0},
              metrics={'policy': 'categorical_accuracy', 'value': 'mse'})

for i in range (1, epochs + 1):
    print ('epoch ' + str (i))
    golois.getBatch (input_data, policy, value, end, groups, i * N)
    history = model.fit(input_data,
                        {'policy': policy, 'value': value}, 
                        epochs=1, batch_size=batch)
    if (i % 5 == 0):
        gc.collect ()
    if (i % 10 == 0):
        golois.getValidation (input_data, policy, value, end)
        val = model.evaluate (input_data,
                              [policy, value], verbose = 0, batch_size=batch)
        print ("val =", val)

model.save ('mbse_corr_100ep_0005.h5')