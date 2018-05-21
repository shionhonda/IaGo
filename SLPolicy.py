import chainer
import chainer.functions as F
import chainer.links as L

import warnings
warnings.filterwarnings('ignore')

class Block(chainer.Chain):
    '''Convolution and ReLU'''
    def __init__(self, out_channels, ksize, pad=1):
        super(Block, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, out_channels, ksize, pad=pad)

    def __call__(self, x):
        h = self.conv(x)
        return F.relu(h)

class SLPolicyNet(chainer.Chain):
    '''Block and dropout'''
    def __init__(self, class_labels=10):
        ksize = 3
        super(SLPolicyNet, self).__init__()
        with self.init_scope():
            self.block1 = Block(64, ksize)
            self.block2 = Block(64, ksize)
            self.block3 = Block(128, ksize)
            self.block4 = Block(128, ksize)
            self.block5 = Block(256, ksize)
            self.block6 = Block(256, ksize)
            self.block7 = Block(256, ksize)
            self.block8 = Block(256, ksize)
            self.fc1 = L.Linear(None, 128, nobias=True)
            self.fc2 = L.Linear(None, 64, nobias=True)

    def __call__(self, x):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.block7(h)
        h = self.block8(h)
        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        return h
