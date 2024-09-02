import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 判断是否有GPU
batch_size = 512  # 设置包的大小
# 下面利用函数DataLoader()下载手写数字图像数据集，
# 下载后默认存放在./data/MNIST子目录下
# 下载训练集，6万张图片，同时打包，包的大小由参数batch_size确定
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,  # 指定下载到./data目录下
                   transform=transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=batch_size, shuffle=True)
# 下载测试集，1万张图片，同时打包，包的大小由参数batch_size确定
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False,
                   transform=transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.1307,), (0.3081,))
                                                 ])),
    batch_size=batch_size, shuffle=True)


# ---------------------------------------------------------------
# 定义卷积神经网络
class Example4_1(nn.Module):
    def __init__(self):
        super().__init__()
        # 下面创建两个卷积层：
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        # 下面创建两个全连接层：
        self.fc1 = nn.Linear(2000, 500)  # 第一个全连接层
        self.fc2 = nn.Linear(500, 10)  # 第二个全连接层

    def forward(self, x):  # x为输入进来的数据包，其形状为batch×1×28×28
        out = self.conv1(x)  # 将x输入第一个卷积层
        out = torch.relu(out)
        out = torch.max_pool2d(out, 2, 2)
        out = self.conv2(out)
        out = torch.relu(out)
        out = out.view(x.size(0), -1)  # 对out扁平化后，以送入全连接层
        out = self.fc1(out)  # 将x输入第一个全连接层
        out = torch.relu(out)
        out = self.fc2(out)  # 将x输入第二个全连接层
        return out


start = time.time()  # 计时开始
example4_1 = Example4_1().to(device)  # 将实例创建在GPU中，如果有的话（下同）
optimizer = optim.Adam(example4_1.parameters())

for epoch in range(20):  # 迭代20代
    example4_1.train()
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)  # 将x和y保存到GPU中
        pre_y = example4_1(x)  # 输入数据包x后
        loss = nn.CrossEntropyLoss()(pre_y, y)  # 使用交叉熵损失函数
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向计算梯度
        optimizer.step()  # 更新参数
        if i % 30 == 0:  # 每计算30个包，做一次数据采样，以显示进度信息
            print(r'训练的代数：{} [当前代完成进度：{} / {} ({:.0f}%)]，当前损失函数值: {:.6f}'.format(epoch, i * len(x),
                                                                                                    len(train_loader.dataset),
                                                                                                    100. * i / len(
                                                                                                        train_loader),
                                                                                                    loss.item()))
# -------------------------------------------
torch.save(example4_1, 'example4_1.pth')  # 保存训练好的模型
example4_1 = torch.load('example4_1.pth')  # 读取保存于磁盘上已经训练好的模型
# 测试---------------------
example4_1.eval()
correct = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pre_y = example4_1(x)  # 输入数据包x，pre_y为其预测输出
        pre_y = torch.argmax(pre_y, dim=1)
        t = (pre_y == y).long().sum()
        correct += t  # 这时t已为被准确预测的样本数
end = time.time()  # 计时结束
print('运行时间：{:.1f}秒'.format(end - start))  # 程序运行时间
correct = correct.data.cpu().item()  # 将保存于GPU中的数据读取到CPU存储器上
correct = 1. * correct / len(test_loader.dataset)  # 计算准确率
print('在测试集上的预测准确率：{:0.2f}%'.format(100 * correct))
