import os.path

import matplotlib.pyplot as plt
# y = [1, 4, 9, 16, 25,36,49, 64]
#
# fig = plt.figure(dpi = 128,figsize = (10,6))
# ax = fig.add_axes([0,0,1,1])
# #使用简写的形式color/标记符/线型
# l1 = ax.plot(x1,y,'ys-')
# l2 = ax.plot(x2,y,'go--')
# ax.legend(labels = ('tv', 'Smartphone'), loc = 'best') # legend placed at lower right
# ax.set_title("Advertisement effect on sales")
# ax.set_xlabel('medium')
# ax.set_ylabel('sales')
# plt.show()

Ux1 = [0.0227,1.8713,3.8285,6.5266,8.2801,9.6878,12.4071,15.1359]
Ux2 = [0.0244,1.8740,3.8355,6.5223,8.2797,9.6987,12.4206,15.1575]
x1 = [1, 16, 30, 42,55, 68, 77,88]
x2 = [1,6,12,18,28, 40, 52, 65]

draw(x1,x2,'loss')

# fig = plt.figure (dpi=100, figsize=(10, 6))
# plt.plot(Ux1,Ux2,c="red",linestyle=':',label='x')
# plt.plot(x1,x2,c="blue",linestyle='-',label='y')
# plt.xlabel(u'λ')
# plt.ylabel(u'Pruned Percentage & Accuracy (%)')
# # plt.plot(lambda1, flops, c='blue', marker='o', linestyle=':', label='FLOPs')
# # plt.plot(lambda1, accuracy, c='red', marker='*', linestyle='-', label='Accuracy')
# plt.legend()
# plt.title("display_test")
# plt.show()

# plt.figure(dpi = 100,figsize = (10,10))
# plt.plot(x1,c='red',linestyle='-',label='loss_tra')
# plt.plot(x2,c='blue',linestyle=':',label='loss_val')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.title('loss_train and loss_eval')
# plt.legend()
# plt.show()