__data__ = "2024/4/1"
__author__ = "张小鹏"


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

n_epochs = 10
batch_size_train = 64
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)


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


def getPicture():
    listpicture1, listpicture2 = [], []
    for i in range(0, 10):
        listpicture1.append('./style' + str(i) + '.png')
        listpicture2.append('./style2' + str(i) + '.png')
    return listpicture1, listpicture2


if __name__ == "__main__":
    # train_and_save_model()
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


