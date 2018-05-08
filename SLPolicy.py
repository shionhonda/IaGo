import chainer
import chainer.functions as F
import chainer.links as L


FILTERS_NUM = 50  # フィルター数(k)
ksize = 3

class SLPolicyNet(chainer.Chain):

    def __init__(self, train=True):
        super(SLPolicyNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, FILTERS_NUM, ksize, pad=1)
            self.conv2 = L.Convolution2D(FILTERS_NUM, FILTERS_NUM, ksize, pad=1)
            self.conv3 = L.Convolution2D(FILTERS_NUM, FILTERS_NUM, ksize, pad=1)
            self.conv4 = L.Convolution2D(FILTERS_NUM, FILTERS_NUM, ksize, pad=1)
            self.conv5 = L.Convolution2D(FILTERS_NUM, FILTERS_NUM, ksize, pad=1)
            self.conv6 = L.Convolution2D(FILTERS_NUM, FILTERS_NUM, ksize, pad=1)
            self.conv7 = L.Convolution2D(FILTERS_NUM, FILTERS_NUM, ksize, pad=1)
            self.conv8 = L.Convolution2D(FILTERS_NUM, FILTERS_NUM, ksize, pad=1)
            self.conv9 = L.Convolution2D(FILTERS_NUM, FILTERS_NUM, ksize, pad=1)
            self.conv10 = L.Convolution2D(FILTERS_NUM, 1, 1, nobias=True)
            self.bias11 = L.Bias(shape=(64))

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        h = F.relu(self.conv8(h))
        h = F.relu(self.conv9(h))
        h = self.conv10(h)
        h = F.reshape(h,(-1,64))
        h = self.bias11(h)
        return h