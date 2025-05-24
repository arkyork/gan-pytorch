
# GAN (PyTorch)

PyTorch を用いたシンプルな GAN（敵対的生成ネットワーク）の実装例です。手書き数字画像データセット MNIST を使用し、ノイズから数字画像を生成する学習を行います。

##  特徴

- PyTorch で書かれた基本的な GAN 構造
- MNIST データセットで画像生成
- 毎エポックごとに画像を可視化
- Generator と Discriminator の損失ログ付き


##  実行環境

- Python 3.8+
- PyTorch 
- torchvision
- matplotlib

### インストール例

```bash
pip install torch torchvision matplotlib
````
##  GAN構成の概要

### Generator（生成器）

* 入力：100次元のランダムノイズ
* 出力：28×28ピクセルの画像
* 活性化関数：ReLU, Tanh

### Discriminator（識別器）

* 入力：28×28ピクセルの画像
* 出力：画像が本物かどうかの確率（0〜1）
* 活性化関数：LeakyReLU, Sigmoid



##  損失の記録

各エポックごとに、以下の損失がログ出力されます。

- Generator Loss（Gの損失）: 偽物を「本物らしく」する能力
- Discriminator Loss（Dの損失）: 本物と偽物を正しく見分ける能力


##  出力例

![noise](img/noise.png)
![temp](img/temp.png)
![end](img/end.png)


##  ファイル構成

```
.
├── GAN.py     # メインコード
├── README.md        # このファイル
└── data/            # MNISTデータがダウンロードされる
```



