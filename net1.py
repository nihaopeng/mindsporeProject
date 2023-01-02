import mindspore.nn as nn
import mindspore.numpy
from mindspore.nn import Cell


class myAgg(nn.Cell):
    def __init__(self):
        super(myAgg,self).__init__()

    def construct(self, x):

        return x

if __name__=="__main__":
    x=mindspore.numpy.rand([1,3,32,32])
    module=myAgg()
    y=module(x)
    print(y)