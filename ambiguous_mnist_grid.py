# ambiguous_mnist_grid.py
# 依存: pip install numpy matplotlib scikit-learn torchvision pillow
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from torchvision.datasets import MNIST
from torchvision import transforms
from PIL import Image

# 1) データ読み込み（自動DL）。白地に黒文字へ統一（MNISTは黒地に白文字なので反転）
tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: 1.0 - x)])
train = MNIST(root="./data", train=True, download=True, transform=tfm)
test  = MNIST(root="./data", train=False, download=True, transform=tfm)

# 2) 学習用にサブサンプル（高速化）
rng = np.random.default_rng(0)
idx = rng.choice(len(train.data), size=20000, replace=False)
Xtr = train.data[idx].numpy().reshape(-1, 28*28).astype(np.float32)/255.0
ytr = np.array(train.targets)[idx]

# 3) 1対多ロジスティック回帰（確率出力可・高速）
clf = make_pipeline(StandardScaler(with_mean=False),
                    LogisticRegression(max_iter=200, solver="saga", multi_class="multinomial"))
clf.fit(Xtr, ytr)

# 4) テスト集合の予測確率から「曖昧さ」を定義
Xte = test.data.numpy().reshape(-1, 28*28).astype(np.float32)/255.0
proba = clf.predict_proba(Xte)
maxp = proba.max(axis=1)           # 最大クラス確率（低いほど曖昧）
order = np.argsort(maxp)           # 昇順＝曖昧→明確
topk = 36                          # 抽出枚数（6x6グリッド）
sel = order[:topk]

# 5) グリッド画像を生成
rows, cols = 6, 6
figsize = (cols*1.6, rows*1.6)
fig = plt.figure(figsize=figsize, dpi=150)
for i, idx in enumerate(sel):
    ax = plt.subplot(rows, cols, i+1)
    img = Xte[idx].reshape(28,28)
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    true = int(test.targets[idx])
    pred = int(proba[idx].argmax())
    ax.set_title(f"y={true}, \u02C6y={pred}, p={maxp[idx]:.2f}", fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.savefig("ambiguous_mnist_grid.png", bbox_inches="tight")
print("Saved -> ambiguous_mnist_grid.png")
