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
