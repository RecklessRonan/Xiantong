import torchvision
import torch
from torchvision import datasets, transforms
import os
from PIL import Image

resnet18 = torchvision.models.resnet18(pretrained=True)
resnet18.load_state_dict(torch.load('resnet18.ckpt'))

device  = torch.device('cuda: 1' if torch.cuda.is_available() else 'cpu')
resnet18 = resnet18.to(device)
resnet18.eval()

file_dir = 'tag_dataset/screen'

def read_jpg():
    images = []
    computers = os.listdir(file_dir)
    for computer in computers:
        computer_dir = file_dir + '/' + computer
        dates = os.listdir(computer_dir)
        for date in dates:
            if date.isdigit(): 
                date_dir = computer_dir + '/' + date
                jpgs = os.listdir(date_dir)
                for jpg in jpgs:
                    jpg_dir = date_dir + '/' + jpg
                    if '.jpg' in jpg_dir:
                        images.append(jpg_dir)
    return images


images = read_jpg()
classes = []

images_dirs = open('images_dirs.txt', 'w')
images_classes = open('images_classes.txt', 'w')

for image in images:
    try:
        img = Image.open(image)
        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(device)
        _, predicted = torch.max(resnet18(img).data, 1)
        classes.append(predicted.item())
        images_dirs.write(image + '\n')
        images_classes.write(str(predicted.item()) + '\n')
        print('image: {}, classes: {}'.format(image, predicted.item()))
    except:
        continue

images_dirs.close()
images_classes.close()