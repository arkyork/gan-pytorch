import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ハイパーパラメータ
batch_size = 64
z_dim = 100
lr = 0.0002
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データセット準備
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # -1〜1に正規化
])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 生成モデル
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 28*28),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.net(z)
        return out.view(-1, 1, 28, 28)

# 識別モデル
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# モデル・損失・最適化
G = Generator().to(device)
D = Discriminator().to(device)
loss = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=lr)
opt_D = optim.Adam(D.parameters(), lr=lr)

# 学習ループ（識別→生成の順）
for epoch in range(epochs):
    for real_imgs, _ in loader:
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        # 本物ラベル: 1、偽物ラベル: 0
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # ----- 識別器の学習 -----
        z = torch.randn(batch_size, z_dim, device=device)
        fake_imgs = G(z)
        D_loss = (
            loss(D(real_imgs), real_labels) +
            loss(D(fake_imgs.detach()), fake_labels)
        ) / 2
        opt_D.zero_grad()
        D_loss.backward()
        opt_D.step()

        # ----- 生成器の学習 -----
        z = torch.randn(batch_size, z_dim, device=device)
        fake_imgs = G(z)
        G_loss = loss(D(fake_imgs), real_labels)  # 本物と思わせたい
        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()

    print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {D_loss.item():.4f} | G Loss: {G_loss.item():.4f}")

    # 画像生成（毎エポック）
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            z = torch.randn(16, z_dim, device=device)
            fake_imgs = G(z).cpu()
            grid = fake_imgs.view(16, 28, 28)
            fig, axes = plt.subplots(1, 16, figsize=(16, 1))
            for i, ax in enumerate(axes):
                ax.imshow(grid[i].squeeze(), cmap='gray')
                ax.axis('off')
            plt.show()
