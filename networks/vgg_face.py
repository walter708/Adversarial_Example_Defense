from keras_vggface.vggface import VGGFace

from .base import KerasNetwork


class VGG16(KerasNetwork):
    def __init__(self):
        super().__init__(VGGFace())

    @staticmethod
    def bounds():
        return super().bounds()

    def name():
        return 'VGG16'
