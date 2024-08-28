# imageの処理 (NumPy配列形式に変換)
from PIL import Image
import os, glob # ファイルを扱うためのライブラリ
import numpy as np
from sklearn import model_selection # 交差検証 データをトレーニング用とテスト用に分割するためのライブラリ

classes = ["monkey", "boar", "crow"] # 動物のラベル定義
num_classes = len(classes)
image_size = 50 # 50ピクセルに変換のための定義

X = []
Y = []

for index, cls in enumerate(classes):
  photos_dir = "./" + cls # 画像の読み込み
  files = glob.glob(photos_dir + "/*.jpg") # パターン一致でファイル一覧を取得する
  for i, file in enumerate(files):
    if i >= 200: break
    image = Image.open(file)
    image = image.convert("RGB") # RGBの三色の数字に変換
    image = image.resize((image_size, image_size)) # サイズを50*50に変換
    data = np.asarray(image) # 数字の配列にして代入
    X.append(data)
    Y.append(index)

# TensorFlowが扱いやすいデータ型に揃える
X = np.array(X)
Y = np.array(Y)

# 学習用と評価用のデータ分離する処理
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,Y, X, Y, test_size=0.2, random_state=42)
# xy = (X_train, X_test, y_train, y_test)
np.savez("./animal.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test) # numpyの配列をテキストファイルとして保存