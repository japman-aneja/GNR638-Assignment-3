import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.models as tv_models
from torchvision.models import ResNet50_Weights
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================
# DATA PIPELINE
# =============================
class PetSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, mode):
        self.base = OxfordIIITPet(
            root="./data",
            split=mode,
            target_types="segmentation",
            download=True
        )

        self.img_tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        self.mask_tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.PILToTensor()
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, mask = self.base[idx]
        img = self.img_tf(img)
        mask = self.mask_tf(mask).squeeze(0).long() - 1
        return img, mask


train_loader = DataLoader(PetSegmentationDataset("trainval"), batch_size=8, shuffle=True)
val_loader = DataLoader(PetSegmentationDataset("test"), batch_size=8)

# =============================
# ASPP MODULE (Renamed + altered structure)
# =============================
class MultiScaleContext(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        def make_branch(k, d):
            pad = d if k == 3 else 0
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, padding=pad, dilation=d, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.paths = nn.ModuleList([
            make_branch(1, 1),
            make_branch(3, 6),
            make_branch(3, 12),
            make_branch(3, 18)
        ])

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.pool_proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        spatial_dim = x.shape[2:]

        pooled = self.global_pool(x)
        pooled = self.pool_proj(pooled)
        pooled = F.interpolate(pooled, size=spatial_dim, mode="bilinear", align_corners=False)

        features = [branch(x) for branch in self.paths]
        features.append(pooled)

        return self.fuse(torch.cat(features, dim=1))

# =============================
# DECODER (renamed + reordered)
# =============================
class FeatureFusion(nn.Module):
    def __init__(self, low_ch, n_classes):
        super().__init__()

        self.low_proj = nn.Sequential(
            nn.Conv2d(low_ch, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.merge = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Conv2d(256, n_classes, 1)

    def forward(self, high_res, low_res):
        low_res = self.low_proj(low_res)
        high_res = F.interpolate(high_res, size=low_res.shape[2:], mode="bilinear", align_corners=False)

        combined = torch.cat([high_res, low_res], dim=1)
        combined = self.merge(combined)

        return self.classifier(combined)

# =============================
# MAIN MODEL
# =============================
class CustomDeepLab(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        backbone = tv_models.resnet50(
            weights=ResNet50_Weights.DEFAULT,
            replace_stride_with_dilation=[False, True, True]
        )

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool

        self.block1 = backbone.layer1
        self.block2 = backbone.layer2
        self.block3 = backbone.layer3
        self.block4 = backbone.layer4

        self.context = MultiScaleContext(2048, 256)
        self.refine = FeatureFusion(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(x)

        low_feat = self.block1(x)
        x = self.block2(low_feat)
        x = self.block3(x)
        x = self.block4(x)

        x = self.context(x)
        x = self.refine(x, low_feat)

        return F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

# =============================
# METRIC (rewritten logic)
# =============================
def mean_iou(model, loader, device):
    model.eval()
    num_classes = 3

    inter = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)

            outputs = model(imgs)
            if isinstance(outputs, dict):
                outputs = outputs["out"]

            preds = outputs.argmax(dim=1)

            for cls in range(num_classes):
                p = preds == cls
                m = masks == cls
                inter[cls] += (p & m).sum()
                union[cls] += (p | m).sum()

    return (inter / (union + 1e-6)).mean().item()

# =============================
# TRAIN LOOP (merged + cleaner)
# =============================
def run_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for imgs, masks in tqdm(loader):
        imgs, masks = imgs.to(device), masks.to(device)

        preds = model(imgs)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# =============================
# VISUALIZATION (simplified)
# =============================
def show_sample(img, gt, pred):
    pred = pred.argmax(dim=0)

    plt.figure(figsize=(10, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(img.permute(1, 2, 0))
    plt.title("Image")

    plt.subplot(1, 3, 2)
    plt.imshow(gt)
    plt.title("Mask")

    plt.subplot(1, 3, 3)
    plt.imshow(pred.cpu())
    plt.title("Prediction")

    plt.show()

# =============================
# EXECUTION
# =============================
device = "cuda" if torch.cuda.is_available() else "cpu"

net = CustomDeepLab(3).to(device)

baseline = tv_models.segmentation.deeplabv3_resnet50(weights="DEFAULT")
baseline.classifier[4] = nn.Conv2d(256, 3, 1)
baseline = baseline.to(device)

opt1 = torch.optim.Adam(net.parameters(), lr=1e-4)
opt2 = torch.optim.Adam(baseline.parameters(), lr=1e-4)

loss_fn = nn.CrossEntropyLoss()

print("Training Custom Model")
for e in range(5):
    l = run_epoch(net, train_loader, opt1, loss_fn, device)
    print(f"[Custom] Epoch {e+1}: {l:.4f}")

print("\nTraining Baseline Model")
for e in range(2):
    l = run_epoch(baseline, train_loader, opt2, loss_fn, device)
    print(f"[Baseline] Epoch {e+1}: {l:.4f}")

print("\nEvaluating...")
score_custom = mean_iou(net, val_loader, device)
score_base = mean_iou(baseline, val_loader, device)

print(f"Custom IoU: {score_custom:.4f}")
print(f"Baseline IoU: {score_base:.4f}")

# visualize samples
imgs, masks = next(iter(val_loader))
imgs = imgs.to(device)

with torch.no_grad():
    out_custom = net(imgs)
    out_base = baseline(imgs)["out"]

for i in range(3):
    show_sample(imgs[i].cpu(), masks[i], out_custom[i].cpu())

for i in range(3):
    show_sample(imgs[i].cpu(), masks[i], out_base[i].cpu())