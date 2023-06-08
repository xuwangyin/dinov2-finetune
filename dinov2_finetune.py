import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
from functools import partial
import time
from tqdm import tqdm


import argparse



import sys
sys.path.append('dinov2')

from dinov2.eval.linear import create_linear_input
from dinov2.eval.linear import LinearClassifier
from dinov2.eval.utils import ModelWithIntermediateLayers

parser = argparse.ArgumentParser(description='Fine-tune DINOv2 on ImageNet100')
parser.add_argument('--arch', '-a', metavar='ARCH', default='dinov2_vitb14', choices=['dinov2_vitb14', 'dinov2_vitl14'],
                    help='')
parser.add_argument('--batch-size', '-b', default=128, type=int, metavar='N',
                    help='mini-batch size (default: 128)')
parser.add_argument('--log-dir', default='./', type=str, metavar='PATH',
                    help='path to directory where to log (default: current directory)')
parser.add_argument('--data-dir', required=True, type=str, metavar='PATH',
                    help='path to the dataset')


args = parser.parse_args()
print(args)

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Dino(nn.Module):
    def __init__(self, type):
        super().__init__()
        # get feature model
        model = torch.hub.load(
            "facebookresearch/dinov2", type, pretrained=True
        ).to(device)
        autocast_ctx = partial(
            torch.cuda.amp.autocast, enabled=True, dtype=torch.float16
        )
        self.feature_model = ModelWithIntermediateLayers(
            model, n_last_blocks=1, autocast_ctx=autocast_ctx
        ).to(device)

        with torch.no_grad():
            sample_input = torch.randn(1, 3, 224, 224).to(device)
            sample_output = self.feature_model(sample_input)

        # get linear readout
        out_dim = create_linear_input(
            sample_output, use_n_blocks=1, use_avgpool=True
        ).shape[1]
        self.classifier = LinearClassifier(
            out_dim, use_n_blocks=1, use_avgpool=True, num_classes=100
        ).to(device)

    def forward(self, x):
        x = self.feature_model(x)
        x = self.classifier(x)
        return x



# Define transforms for the training and validation datasets
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Dino(args.arch)

# Disable gradient for feature model
for param in model.feature_model.parameters():
    param.requires_grad = False
    
for param in model.classifier.parameters():
    param.requires_grad = True

# Define loss function
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Train the model
num_epochs = 25
best_acc = 0.0
for epoch in range(num_epochs):

    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    for phase in ['train', 'val']:
        start_time = time.time()
        if phase == 'train':
            model.feature_model.eval()
            model.classifier.train()
        else:
            model.feature_model.eval()
            model.classifier.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(dataloaders_dict[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            with torch.no_grad():
                features = model.feature_model(inputs)

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model.classifier(features)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders_dict[phase].dataset)

        end_time = time.time()
        time_elapsed = end_time - start_time
        print('{} Loss: {:.4f} Acc: {:.4f} Time: {:.0f}m {:.0f}s'.format(phase, epoch_loss, epoch_acc, time_elapsed // 60, time_elapsed % 60)) # Modify this line
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model.pth'))


    print()

print('Training complete')

