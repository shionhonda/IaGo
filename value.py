import chainer
import chainer.functions as F
import chainer.links as L

class Block(chainer.Chain):
    def __init__(self, out_channels, ksize, pad=1):
        super(Block, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, out_channels, ksize, pad=pad)

    def __call__(self, x):
        h = self.conv(x)
        return F.relu(h)

class ValueNet(chainer.Chain):
    def __init__(self):
        ksize = 3
        super(ValueNet, self).__init__()
        with self.init_scope():
            self.block1 = Block(64, ksize)
            self.block2 = Block(128, ksize)
            self.block3 = Block(128, ksize)
            self.block4 = Block(128, ksize)
            self.block5 = Block(128, ksize)
            self.block6 = Block(128, ksize)
            self.block7 = Block(128, ksize)
            self.block8 = Block(128, ksize)
            self.block9 = Block(1, ksize)
            self.fc10 = L.Linear(None, 128, nobias=True)
            self.fc11 = L.Linear(None, 1, nobias=True)

    def __call__(self, x):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.block7(h)
        h = self.block8(h)
        h = self.block9(h)
        h = self.fc10(h)
        h = F.dropout(h, 0.4)
        h = self.fc11(h).reshape(-1)

        return h
