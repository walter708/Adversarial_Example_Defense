from keras_vggface.vggface import VGGFace
from keras.layers import Dense, Flatten
import tensorflow as tf
from keras import layers, Model
import numpy as np
import keras
import foolbox as fb
from foolbox.criteria import Misclassification
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from skimage.transform import rescale, resize, downscale_local_mean


# from matplotlib.pyplot import imshow
# import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report


def image_proc():
    from sklearn.datasets import fetch_lfw_people
    lfw_people = fetch_lfw_people(
        min_faces_per_person=100, resize=1, color=True)
    images = []
    for image in lfw_people.images:
        image = resize(image, (224, 224, 3))
        images.append(image)

    img = np.array(images)
    img_norm = img/255
    lab = np.array(lfw_people.target)

    return img_norm, lab


def model():
    vgg_x = VGGFace(model='vgg16', weights='vggface',
                    input_shape=(224, 224, 3), include_top=False)
    last_layer = vgg_x.get_layer('pool5').output
    print(type(last_layer))
    x = Flatten(name='flatten')(last_layer)
    x = Dense(4096, activation='relu', name='fc6')(x)
    out = Dense(5, activation='softmax', name='fc8')(x)
    custom_vgg_model = Model(vgg_x.input, out)
    return custom_vgg_model


custom_vgg_model = model()

custom_vgg_model.load_weights(
    r"C:\Users\okpal\Documents\home\levn\workspace\ece650\ECE653 Project  Spring 2021\Project_Random_Based_Method\custom_vgg_weights.h5")

img, lab = image_proc()

preprocessing = dict()
bounds = (0, 1)
fmodel = fb.TensorFlowModel(
    custom_vgg_model, bounds=bounds, preprocessing=preprocessing)
fmodel = fmodel.transform_bounds((0, 1))
assert fmodel.bounds == (0, 1)

labels = tf.convert_to_tensor(lab)
images = tf.convert_to_tensor(img)

fb.utils.accuracy(fmodel, images[:100], labels[:100])
attack = fb.attacks.FGSM()
criterion = Misclassification(labels[:100])
advs, _, is_adv = attack(
    model=fmodel, inputs=images[:100], criterion=criterion, epsilons=0.01)
fb.utils.accuracy(fmodel, advs, labels[:100])
