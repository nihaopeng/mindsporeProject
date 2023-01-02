import mindspore
from mindspore import nn
import mindspore.nn as F

class Net (nn.Cell):
    def __init__(self):
        super (Net, self).__init__ ()
        self.relu=nn.ReLU()
        # 卷积层1.输入是32*32*3，计算（32-5）/ 1 + 1 = 28，那么通过conv1输出的结果是28*28*6
        self.conv1 = nn.Conv2d (3, 6, 5,1,padding = 0,pad_mode = 'pad')  # imput:3 output:6, kernel:5
        # 池化层， 输入时28*28*6， 窗口2*2，计算28 / 2 = 14，那么通过max_poll层输出的结果是14*14*6
        self.pool = nn.MaxPool2d (2, 2)  # kernel:2 stride:2
        # 卷积层2， 输入是14*14*6，计算（14-5）/ 1 + 1 = 10，那么通过conv2输出的结果是10*10*16
        self.conv2 = nn.Conv2d (6, 16, 5,padding = 0,pad_mode = 'pad')  # imput:6 output:16, kernel:5
        # 全连接层1
        self.fc1 = nn.Dense (400, 120)  # input：16*5*5，output：120
        # 全连接层2
        self.fc2 = nn.Dense (120, 84)  # input：120，output：84
        # 全连接层3
        self.fc3 = nn.Dense (84, 10)  # input：84，output：10
        self.flatten=nn.Flatten()

    def construct(self, x):
        # 卷积1
        '''
        32x32x3 --> 28x28x6 -->14x14x6
        '''
        x = self.pool (self.relu (self.conv1 (x)))
        # 卷积2
        '''
        14x14x6 --> 10x10x16 --> 5x5x16
        '''
        x = self.pool (self.relu (self.conv2 (x)))
        # 改变shape
        x=self.flatten(x)
        # 全连接层1
        x = self.relu (self.fc1 (x))
        # 全连接层2
        x = self.relu (self.fc2 (x))
        # 全连接层3

        x = self.fc3 (x)
        return x

if __name__ == "__main__":
    x = mindspore.numpy.rand ([1, 3, 32, 32])
    module = Net ()
    y = module (x)
    print (y)