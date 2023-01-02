import mindspore.nn as nn
import mindspore.numpy
from mindspore.nn import Cell


class myAgg (nn.Cell):
    def __init__(self):
        super (myAgg, self).__init__ ()
        self.relu=nn.ReLU()
        self.c1 = nn.Conv2d (3, 48, 11, 4, padding = 0, pad_mode = "pad")
        # self.c2=nn.Conv2d(96,256,5,2,padding=0,pad_mode="pad")/
        self.pool0=nn.MaxPool2d(3,2)
        self.c2 = nn.Conv2d (48, 128, 5,1,padding = 2, pad_mode = "pad")
        self.pool1 = nn.MaxPool2d (3,2)
        #/
        self.c3 = nn.Conv2d (128, 192, 3, 1, padding = 1, pad_mode = "pad")
        #/
        self.c4 = nn.Conv2d (192, 192, 3, 1, padding = 1, pad_mode = "pad")
        #/
        self.c5 = nn.Conv2d (192, 128, 3, 1, padding = 1, pad_mode = "pad")
        #/
        # self.c6 = nn.Conv2d (384, 256, 3, 1, padding = 0, pad_mode = "pad")
        # #/
        self.pool3=nn.MaxPool2d(3,2)

        self.classfier=nn.SequentialCell([
            nn.Flatten (),
            nn.Dense (4608,2048),
            nn.Dropout (),
            nn.Dense (2048, 2048),
            nn.Dropout (),
            nn.Dense (2048, 1000),
            nn.Dropout (),
            nn.Dense (1000, 10)
        ])

    def construct(self, x):
        x=self.relu(self.c1(x))
        x=self.pool0(x)
        x=self.relu(self.c2(x))
        x=self.pool1(x)
        x=self.relu(self.c3(x))
        x=self.relu(self.c4(x))
        x=self.relu(self.c5(x))
        x=self.pool3(x)
        x=self.classfier(x)
        return x


if __name__ == "__main__":
    x = mindspore.numpy.rand ([1, 3, 227, 227])
    module = myAgg ()
    y = module (x)
    print (y)
