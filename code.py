import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt

img_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
mask_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.PILToTensor()])

class PET_dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.dataset = OxfordIIITPet(root='./data', split=split, target_types='segmentation', download=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        img = img_transform(img)
        mask = mask_transform(mask)
        mask = mask.squeeze(0)
        mask = mask.long() - 1
        return img, mask

trainLoader = DataLoader(PET_dataset("trainval"), batch_size=8, shuffle=True)
valLoader = DataLoader(PET_dataset("test"), batch_size=8)

class ASPP(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(inc, outc, 1, bias=False), nn.BatchNorm2d(outc), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(nn.Conv2d(inc, outc, 3, padding=6, dilation=6, bias=False), nn.BatchNorm2d(outc), nn.ReLU(inplace=True))
        self.b3 = nn.Sequential(nn.Conv2d(inc, outc, 3, padding=12, dilation=12, bias=False), nn.BatchNorm2d(outc), nn.ReLU(inplace=True))
        self.b4 = nn.Sequential(nn.Conv2d(inc, outc, 3, padding=18, dilation=18, bias=False), nn.BatchNorm2d(outc), nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pool_conv = nn.Sequential(nn.Conv2d(inc, outc, 1, bias=False), nn.BatchNorm2d(outc), nn.ReLU(inplace=True))
        self.out = nn.Sequential(nn.Conv2d(outc * 5, outc, 1, bias=False), nn.BatchNorm2d(outc), nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[2:]
        p = self.pool(x)
        p = self.pool_conv(p)
        p = F.interpolate(p, size=size, mode='bilinear', align_corners=False)
        x1 = self.b1(x)
        x2 = self.b2(x)
        x3 = self.b3(x)
        x4 = self.b4(x)
        x = torch.cat([x1, x2, x3, x4, p], dim=1)
        return self.out(x)

class Decoder(nn.Module):
    def __init__(self, low_channels, num_classes):
        super().__init__()
        self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.final = nn.Conv2d(256, num_classes, 1)
    def forward(self, high, low):
        low = self.reduce(low)
        high = F.interpolate(high, size=low.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([high, low], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.final(x)

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT, replace_stride_with_dilation=[False, True, True])
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.aspp = ASPP(2048, 256)
        self.decoder = Decoder(256, num_classes)
    def forward(self, x):
        x = self.layer0(x)
        x = self.maxpool(x)
        low = self.layer1(x)
        x = self.layer2(low)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.aspp(x)
        x = self.decoder(x, low)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        return x

def compute_dataset_iou(model, loader, device):
    intersection = torch.zeros(3, device=device)
    union = torch.zeros(3, device=device)
    model.eval()
    with torch.no_grad():
        for img, mask in loader:
            img = img.to(device)
            mask = mask.to(device)
            pred = model(img)
            if isinstance(pred, dict):
                pred = pred['out']
            pred = torch.argmax(pred, dim=1)
            for c in range(3):
                pred_c = pred == c
                mask_c = mask == c
                intersection[c] += (pred_c & mask_c).sum()
                union[c] += (pred_c | mask_c).sum()
    iou = intersection / (union + 1e-6)
    return iou.mean().item()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DeepLabV3Plus(3).to(device)
official_model = models.segmentation.deeplabv3_resnet50(weights="DEFAULT")
official_model.classifier[4] = nn.Conv2d(256, 3, 1)
official_model = official_model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer_official = torch.optim.Adam(official_model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

def train(model):
    model.train()
    total_loss = 0
    for img, mask in tqdm(trainLoader):
        img = img.to(device)
        mask = mask.to(device)
        pred = model(img)
        loss = criterion(pred, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(trainLoader)

def train_official():
    official_model.train()
    total_loss = 0
    for img, mask in tqdm(trainLoader):
        img = img.to(device)
        mask = mask.to(device)
        pred = official_model(img)['out']
        loss = criterion(pred, mask)
        optimizer_official.zero_grad()
        loss.backward()
        optimizer_official.step()
        total_loss += loss.item()
    return total_loss / len(trainLoader)

print("Training DeepLabV3+...")
for epoch in range(5):
    loss = train(model)
    print(f"My Model Epoch {epoch+1}: {loss:.4f}")

print("\nTraining Official Model...")
for epoch in range(2):
    loss = train_official()
    print(f"Official Epoch {epoch+1}: {loss:.4f}")

print("\nEvaluating...")
my_iou = compute_dataset_iou(model, valLoader, device)
official_iou = compute_dataset_iou(official_model, valLoader, device)
print(f"My Model IoU: {my_iou:.4f}")
print(f"Official Model IoU: {official_iou:.4f}")
