# Description
# 这是学习界的hello world #
# 在本次报告中,将使用pytorch学习框架去训练识别手写体字体 #
# 1.导入必要的文件
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np

# 下面我们来一行行解读一下

# epoch是我们的训练次数,这个次数过少会造成训练模型的不够精确,训练次数过多需要更多的时间,而且也有可能过拟合
n_epochs = 10

# 这个是分批测试的意思
batch_size_train = 64

# 学习率
'''
在模型训练的优化部分，调整最多的一个参数就是学习率，合理的学习率可以使优化器快速收敛。 
一般在训练初期给予较大的学习率，随着训练的进行，学习率逐渐减小
'''
learning_rate = 0.01

# 这些也是超参数
momentum = 0.5

# 间隔
log_interval = 10

# 随机化
random_seed = 1
torch.manual_seed(random_seed)

# 这里是进行GPU加速的部分,如果有可用的GPU的话,那么直接进行加速即可
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''torch.utils.data.DataLoader(...)： 这是 PyTorch 
中用于加载数据并以批量形式提供给模型训练的数据加载器类的构造函数。它封装了许多与数据处理相关的实用功能，如批次生成、数据混洗（shuffle）、多进程加载等。

torchvision.datasets.MNIST('./data/', train=True, download=True, ...)： 这一部分指定了要使用的数据集。在这里，使用的是 
torchvision.datasets.MNIST，这是一个内置的 PyTorch 数据集，包含了 MNIST 手写数字数据。参数说明如下：

./data/: 数据集保存的根目录。如果本地没有该数据集，指定此路径会触发自动下载。(你知道我为什么不着急让你下载了吧)
train=True: 表明加载的是 MNIST 数据集的训练集部分。
download=True: 如果所需的文件在指定目录下未找到，设置为 True 会让程序自动从互联网上下载数据集。
transform=torchvision.transforms.Compose([...])： 这部分定义了对原始图像数据进行预处理的一系列变换。Compose 类将多个变换组合成一个序列，按顺序依次应用到每个图像上。这里有两个变换：

torchvision.transforms.ToTensor(): 将图像从 PIL 图像格式或 numpy 数组转换为 PyTorch 张量（Tensor），同时将像素值从 0 到 255 范围内线性映射到 0.0 到 1.0 
之间。 torchvision.transforms.Normalize((0.1307,), (0.3081,)): 对图像张量进行标准化处理，使其均值（mean）变为 0.1307，标准差（std）变为 0.3081。这是针对 
MNIST 数据集统计得出的特定均值和标准差，目的是使数据具有零均值和单位方差，有助于加快模型收敛和提高泛化能力。 batch_size=batch_size_train： 指定每个批次加载的样本数量。batch_size_train 
应该是一个先前定义好的整数，就是上文中的64,也就是一次训练64张图片

shuffle=True： 设置为 True 意味着在每个 epoch 开始时，数据加载器会对整个训练集进行随机排序，这样模型在训练过程中会以不同的顺序接触到样本，有助于打破相关性，提高模型的泛化能力和训练效果。
'''

# 总之,经过一系列变化,我们拿到了可用的数据
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)


'''
下面这段逻辑是核心
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


network = Net().to(device)
save_path = './model.pth'


def train_and_save_model():
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    for epoch in range(1, n_epochs + 1):
        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                               len(train_loader.dataset),
                                                                               100. * batch_idx / len(train_loader),
                                                                               loss.item()))
                torch.save(network.state_dict(), save_path)


# 读取训练好的网络参数
net = Net().to(device)
net.eval()
a = torch.load('./model.pth')
net.load_state_dict(torch.load('./model.pth'))


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))  # Resize to 28x28
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    img = (img - 0.1307) / 0.3081  # Normalize using MNIST statistics
    img = img.reshape(1, 1, 28, 28)  # Add batch and channel dimensions
    return torch.tensor(img).to(device)


'''
这就是我自己添加的一个方法,目的是去对照遍历一下我们的结果
'''
def getPicture():
    listpicture1, listpicture2 = [], []
    for i in range(0, 10):
        listpicture1.append('./style' + str(i) + '.png')
        listpicture2.append('./style2' + str(i) + '.png')
    return listpicture1, listpicture2


if __name__ == "__main__":
    list_ans1, list_ans2 = getPicture()
    i = 0
    target = 0
    miss = 0
    for pic_str1, pic_str2 in zip(list_ans1, list_ans2):
        img1 = preprocess_image(pic_str1)
        target += 1
        img2 = preprocess_image(pic_str2)
        target += 1
        outputs1 = net(img1)
        outputs2 = net(img2)
        _, predicted1 = torch.max(outputs1.data, 1)
        _, predicted2 = torch.max(outputs2.data, 1)
        print('-------------------------------------------------------------------------')
        print(f'Predicted digit for {pic_str1}: {predicted1.to("cpu").numpy().squeeze()}')
        print(f'Predicted digit for {pic_str2}: {predicted2.to("cpu").numpy().squeeze()}')
        if i != predicted1.to("cpu").numpy().squeeze():
            print('发生了一次错误匹配')
            print('匹配失败的是第一批图片中的数字', i)
            miss +=1
        if i != predicted2.to("cpu").numpy().squeeze():
            print('发生了一次错误匹配')
            print('匹配失败的是第二批图片中的数字', i)
            miss+=1
        i+=1

    print('总共匹配的次数是:', target)
    print('发生miss的次数是:', miss)
    print('成功的次数是: ', target - miss)
    print('在20张数字图片中，准确率为:' , 1 - miss / target)


