import seaborn as sns
import sklearn


import matplotlib.pyplot as plt

fusion = 'C:\\Users\\Administrator\\Desktop\\newTran\\tecstucompare\\fusion1.txt'
stu = 'C:\\Users\\Administrator\Desktop\\newTran\\tecstucompare\\stu1.txt'
tec = 'C:\\Users\\Administrator\\Desktop\\newTran\\tecstucompare\\tec1.txt'

fusionfile = open(fusion, 'r',encoding='utf-8')
stufile = open(stu, 'r',encoding='utf-8')
tecfile = open(tec, 'r',encoding='utf-8')

fusionAcc = []
stuAcc = []
tecAcc = []

stuVal = []
fusionVal = []
tecVal = []

while True:
    line = fusionfile.readline()
    if not line:
        break
    if not line.__contains__("val_accuracy"):
        continue
    if line.__contains__("Epoch"):
        continue
    if line.__contains__("val_f1:"):
        continue

    print(line, end='')
    accIndex = line.index("val_loss")
    acc = line[accIndex+9:accIndex+16]
    fusionAcc.append(float(acc))
    accValIndex = line.index("val_accuracy")
    accVal = line[accValIndex + 13:accValIndex + 20]
    fusionVal.append(float(accVal))
fusionfile.close()

while True:
    line = stufile.readline()
    if not line:
        break
    if not line.__contains__("val_accuracy"):
        continue
    if line.__contains__("Epoch"):
        continue
    if line.__contains__("val_f1:"):
        continue

    print(line, end='')
    accIndex = line.index("val_loss")
    acc = line[accIndex + 9:accIndex + 16]
    stuAcc.append(float(acc))
    accValIndex = line.index("val_accuracy")
    accVal = line[accValIndex + 13:accValIndex + 20]
    stuVal.append(float(accVal))
stufile.close()

while True:
    line = tecfile.readline()
    if not line:
        break
    if not line.__contains__("val_accuracy"):
        continue
    if line.__contains__("Epoch"):
        continue
    if line.__contains__("val_f1:"):
        continue

    print(line, end='')
    accIndex = line.index("val_loss")
    acc = line[accIndex + 9:accIndex + 16]
    tecAcc.append(float(acc))
    accValIndex = line.index("val_accuracy")
    accVal = line[accValIndex + 13:accValIndex + 20]
    tecVal.append(float(accVal))
tecfile.close()

x= range(1,101) #创建等差数列(0,2)分成100份
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']#设置字体为SimHei显示中文
plt.plot(x,fusionAcc,color="red", linewidth=2,label='DF-Model')#（x,x平方）坐标画图
plt.plot(x,stuAcc,linewidth=2, color="orange",label='Learning Network')#（x,x三次方）坐标画图
plt.plot(x,tecAcc,color='blue',linewidth=2,label='Guiding Network')#（x,x三次方）坐标画图
# plt.ylim(0.85, 1)  # 设置Y轴的范围从0到20
plt.xlabel('Epoch')#x坐标轴名
plt.ylabel('Loss')#y坐标轴名
plt.title('Test Loss')
plt.legend()#加上图例
plt.grid()
plt.savefig('loss.png')
# plt.savefig('acc.eps')
plt.savefig('loss.pdf')

plt.show()#显示图像
# plt.figure(figsize=(20, 10))  # 宽度为10英寸，高度为6英寸
x= range(1,101) #创建等差数列(0,2)分成100份
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']#设置字体为SimHei显示中文
plt.plot(x,fusionVal,color="red", linewidth=2, label='DF-Model ')#（x,x平方）坐标画图
plt.plot(x,stuVal, linewidth=2, color="orange", label='Learning network')#（x,x三次方）坐标画图
plt.plot(x,tecVal,color='blue',linewidth=2,label='Guiding Network')#（x,x三次方）坐标画图
plt.ylim(0.8, 0.97)  # 设置Y轴的范围从0到20

plt.xlabel('Epoch')#x坐标轴名
plt.ylabel('Accuracy')#y坐标轴名
plt.title('Test Accuracy')
plt.legend()#加上图例
plt.grid()
plt.savefig('test.png')
# plt.savefig('loss.eps')
plt.savefig('test.pdf')
plt.show()#显示图像
