import os
import time
import matplotlib.pyplot as plt
import mindspore
from mindspore.dataset import Cifar10Dataset, vision
from mindspore.dataset.vision import transforms
from mindspore.dataset import transforms
from mindspore import dtype as mstype, nn, ops
from mindspore.nn import optim

from net2 import Net

train_dataset=Cifar10Dataset("./data/cifar-10-binary/cifar-10-batches-bin",shuffle=True,usage="train")
test_dataset=Cifar10Dataset("./data/cifar-10-binary/cifar-10-batches-bin",shuffle=True,usage="test")

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

module=Net()

loss_fn=nn.SoftmaxCrossEntropyWithLogits(sparse=True,reduction="mean")

optimizer=optim.SGD(module.trainable_params(),learning_rate=0.001,momentum=0.9)

epoch=100


def train(train_set,loss_fn,optimizer,module):
    def forward_fn(image,label):
        output=module(image)
        loss=loss_fn(output,label)
        return loss,output
    grad_fn=ops.value_and_grad(forward_fn,None,optimizer.parameters,True)
    #datalen=train_set.get_dataset_size()
    module.set_train()
    acc_train=0.0
    loss_train=0.0
    total=0
    for batch,(image,label) in enumerate(train_set):
        #print(image.shape)
        output=module(image)
        (loss,_),grads=grad_fn(image,label)
        loss_train+=loss_fn(output,label)
        ops.depend(loss,optimizer(grads))
        argmax=ops.ArgMaxWithValue(1)
        pred,_=argmax(output)
        acc=(label==pred).asnumpy().sum()/len(label)
        if (batch+1)%300==0 :
            print(_)
            print(pred)
            print(label)
            print("batch{},acc={},sum={},label={}".format(batch+1, acc, (label==pred).asnumpy().sum(),len(label)))
        acc_train+=acc
        total+=1
    print("acc_train={}".format(acc_train/total))
    return acc_train/total,loss_train/total

def test(test_set,module):
    acc_test=0.0
    loss_test=0.0
    total=0
    module.set_train(False)
    for batch,(image,label) in enumerate(test_set):
        output=module(image)
        loss_test+=loss_fn(output,label)
        argmax=ops.ArgMaxWithValue(1)
        #correctsum=ops.reduce_sum()
        pred,_=argmax(output)
        acc=(label==pred).asnumpy().sum()/len(label)
        acc_test+=acc
        total+=1
    print("acc_test={}".format(acc_test/total))
    return acc_test/total,loss_test/total

def draw(tra,val,title):
    plt.figure (dpi = 100, figsize = (10, 6))
    plt.plot (tra, c = 'red', linestyle = '-', label = title+'_tra')
    plt.plot (val, c = 'blue', linestyle = ':', label = title+'loss_val')
    plt.xlabel ('epoch')
    plt.ylabel ('loss')
    plt.title (title)
    plt.legend ()
    if not os.path.exists ("comparePic"):
        os.mkdir ("comparePic")
    plt.savefig ('comparePic/alexnet_loss.png')
    plt.show ()


loss_tra=[]
acc_tra=[]
loss_val=[]
acc_val=[]

time_start = time.clock()
max_acc=0.0
for i in range(epoch):
    print("-------------------\nepoch{}".format(i+1))
    acc_train,loss_train=train(train_dataset,loss_fn,optimizer,module)
    acc_test,loss_test=test(test_dataset,module)
    loss_tra.append(loss_train)
    acc_tra.append(acc_train)
    loss_val.append(loss_test)
    acc_val.append(acc_test)
    if acc_test>max_acc:
        max_acc=acc_test
        if not os.path.exists("save_modules"):
            os.mkdir("save_modules")
        mindspore.save_checkpoint(module,"save_modules/best_module_to_classfy_alexnet.ckpt")
        print("save the better module")

time_end=time.clock()

print("use_time={}".format(time_end-time_start))
draw(loss_tra,loss_val,'loss')
draw(acc_tra,acc_val,'acc')
