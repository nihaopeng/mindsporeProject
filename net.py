import mindspore.nn as nn
import mindspore.numpy
from mindspore.nn import Cell


class myVgg(nn.Cell):
    def __init__(self):
        super(myAgg,self).__init__()
        self.c1=nn.SequentialCell([
            nn.Conv2d(3,64,3,1,padding=1,pad_mode='pad'),
            nn.BatchNorm2d(64,affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        ])
        self.c2=nn.SequentialCell([
            nn.Conv2d(64, 128, 3, 1, padding=1,pad_mode='pad'),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ])
        self.c3=nn.SequentialCell([
            nn.Conv2d(128,256,3,1,padding=1,pad_mode='pad'),
            nn.BatchNorm2d(256,affine=True),
            nn.ReLU(),
            nn.Conv2d(256,256,3,1,padding=1,pad_mode='pad'),
            nn.BatchNorm2d(256,affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        ])
        self.c4 = nn.SequentialCell([
            nn.Conv2d(256, 512, 3, 1, padding=1,pad_mode='pad'),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, padding=1,pad_mode='pad'),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ])
        self.c5 = nn.SequentialCell([
            nn.Conv2d(512, 512, 3, 1, padding=1,pad_mode='pad'),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, padding=1,pad_mode='pad'),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ])
        self.classifier=nn.SequentialCell([
            nn.Flatten(),
            nn.Dense(512,64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Dense(64,64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Dense(64,10),
            #nn.Softmax()
        ])
    def construct(self, x):
        x=self.c1(x)
        x=self.c2(x)
        x=self.c3(x)
        x=self.c4(x)
        x=self.c5(x)
        print(x.shape)
        x=self.classifier(x)

        return x

if __name__=="__main__":
    x=mindspore.numpy.rand([1,3,32,32])
    module=myVgg()
    y=module(x)
    print(y)