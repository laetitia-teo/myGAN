from dataset import MnistDataset
from keras_gan import GAN

mnist = MnistDataset(64)
gan = GAN(mnist)
