from dataset import MnistDataset
from model import GAN

mnist = MnistDataset(64)
gan = GAN(mnist)
