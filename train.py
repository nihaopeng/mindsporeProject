import os.path
import time
from mindspore import dtype as mstype
import mindspore
import numpy as np
from mindspore import nn, ops
from mindspore.dataset import Cifar10Dataset, vision
from mindspore.dataset.vision import transforms
from mindspore.dataset import transforms
from mindspore.nn import optim

from net import myVgg


train_dataset=Cifar10Dataset("./data/cifar-10-binary/cifar-10-batches-bin",shuffle=True,usage="train")
test_dataset=Cifar10Dataset("./data/cifar-10-binary/cifar-10-batches-bin",shuffle=True,usage="test")

print(train_dataset.get_col_names())

def datapipe(dataset,batchsize):

    image_transforms=transforms.Compose([
        vision.Resize([32,32]),
        #vision.ToTensor(),
        vision.Normalize(mean=(0.4914,0.4822,0.4465),std=(0.2023,0.1994,0.2010)),
        #ops.L2Normalize(),
        vision.HWC2CHW()
    ])
    lable_transforms=transforms.TypeCast(mstype.int32)
    dataset=dataset.map(image_transforms,"image")
    dataset=dataset.map(lable_transforms,"label")
    dataset=dataset.batch(batchsize)
    return dataset

train_dataset=datapipe(train_dataset,16)
test_dataset=datapipe(test_dataset,16)

mindspore.set_context(device_target="GPU")

module=myVgg()

loss_fn=nn.SoftmaxCrossEntropyWithLogits(sparse=True,reduction="mean")

optimizer=optim.SGD(module.trainable_params(),learning_rate=0.01,momentum=0.9)

epoch=100

def train(train_set,loss_fn,optimizer,module):

    max_acc=0.0
    def forward_fn(image,label):
        output=module(image)
        loss=loss_fn(output,label)
        return loss,output
    grad_fn=ops.value_and_grad(forward_fn,None,optimizer.parameters,True)
    #datalen=train_set.get_dataset_size()
    module.set_train()
    acc_train=0.0
    total=0
    for batch,(image,label) in enumerate(train_set):
        #print(image.shape)
        output=module(image)
        (loss,_),grads=grad_fn(image,label)
        ops.depend(loss,optimizer(grads))
        argmax=ops.ArgMaxWithValue(1)
        #correctsum=ops.reduce_sum()
        pred,_=argmax(output)
        #print(pred)
        acc=(label==pred).asnumpy().sum()/len(label)
        if (batch+1)%300==0 :
            print(_)
            print(pred)
            print(label)
            print("batch{},acc={},sum={},label={}".format(batch+1, acc, (label==pred).asnumpy().sum(),len(label)))
        acc_train+=acc
        total+=1
    print("acc_train={}".format(acc_train/total))

def test(test_set,module):
    acc_test=0.0
    total=0
    module.set_train(False)
    for batch,(image,label) in enumerate(test_set):
        output=module(image)
        argmax=ops.ArgMaxWithValue(1)
        #correctsum=ops.reduce_sum()
        pred,_=argmax(output)
        acc=(label==pred).asnumpy().sum()/len(label)
        acc_test+=acc
        total+=1
    print("acc_test={}".format(acc_test/total))
    return acc_test/total


time_start = time.time()
max_acc=0.0
for i in range(epoch):
    print("-------------------\nepoch{}".format(i+1))
    train(train_dataset,loss_fn,optimizer,module)
    acc=test(test_dataset,module)
    if acc>max_acc:
        max_acc=acc
        if not os.path.exists("save_modules"):
            os.mkdir("save_modules")
        mindspore.save_checkpoint(module,"save_modules/best_module_to_classfy.ckpt")
        print("save the better module")

time_end=time.time()
print("use_time={}".format(time_end-time_start))