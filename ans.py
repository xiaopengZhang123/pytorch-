import torch
from main import Net
from PIL import Image
import torchvision.transforms as transforms
# 加载模型
model = Net()
model.load_state_dict(torch.load('model.pth'))
model.eval()
preprocess = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

image_path = './style6.jpg'  # 替换为您的手写数字图像路径
image = Image.open(image_path).convert('L')  # 打开并转换为灰度图像
input_tensor = preprocess(image).unsqueeze(0)  # 转换为 Tensor 并添加 batch 维度

with torch.no_grad():
    output = model(input_tensor)
    prediction = output.argmax(dim=1).item()

print(f"Predicted digit: {prediction}")

