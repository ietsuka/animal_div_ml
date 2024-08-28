# imageの処理 (NumPy配列形式に変換)
from PIL import Image
import os, glob # ファイルを扱うためのライブラリ
import numpy as np
from sklearn import model_selection # 交差検証 データをトレーニング用とテスト用に分割するためのライブラリ

classes = ["monkey", "boar", "crow"] # 動物のラベル定義
num_classes = len(classes)
image_size = 50 # 50ピクセルに変換のための定義
num_testdata = 100

X_train = []
X_test = []
Y_train = []
Y_test = []

for index, cls in enumerate(classes):
  photos_dir = "./" + cls # 画像の読み込み
  files = glob.glob(photos_dir + "/*.jpg") # パターン一致でファイル一覧を取得する
  for i, file in enumerate(files):
    if i >= 200: break
    image = Image.open(file)
    image = image.convert("RGB") # RGBの三色の数字に変換
    image = image.resize((image_size, image_size)) # サイズを50*50に変換
    data = np.asarray(image) # 数字の配列にして代入

    if i < num_testdata:
      X_test.append(data)
      Y_test.append(index)
    else:
      for angle in range(-20,20,5):
        # 回転
        img_r = image.rotate(angle)
        data = np.asarray(img_r)
        X_train.append(data)
        Y_train.append(index)

        # 反転
        img_trans = img_r.transpose(Image.FLIP_LEFT_RIGHT)
        data = np.asarray(img_trans)
        X_train.append(data)
        Y_train.append(index)

X_train  = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(Y_train)
y_test = np.array(Y_test)

np.savez("./animal_aug.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test) # numpyの配列をテキストファイルとして保存