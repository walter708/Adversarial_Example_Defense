from keras_vggface.vggface import VGGFace
from keras.layers import Dense, Flatten
import tensorflow as tf
from keras import layers, Model

# img, lab = image_proc()

vgg_x = VGGFace(model='vgg16', weights='vggface',
                input_shape=(224, 224, 3), include_top=False)
last_layer = vgg_x.get_layer('pool5').output
print(type(last_layer))
x = Flatten(name='flatten')(last_layer)
x = Dense(4096, activation='relu', name='fc6')(x)
out = Dense(5, activation='softmax', name='fc8')(x)
model = Model(vgg_x.input, out)

model.load_weights(
    r"C:\Users\okpal\Documents\home\levn\workspace\ece650\ECE653 Project  Spring 2021\Project_Random_Based_Method\custom_vgg_weights.h5")
