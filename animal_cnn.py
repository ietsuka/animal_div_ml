from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
import numpy as np
import tensorflow

classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50

# トレーニングを実行するメインの関数を定義する
def main():
  # X_train, X_test, y_train, y_test = np.load("./animal.npy",allow_pickle=True) # ファイルからデータを配列に読み込む
  loaded_data = np.load("./drive/MyDrive/ML/image_div/animal.npz")

  # 各配列にアクセス
  X_train = loaded_data["X_train"]
  X_test = loaded_data["X_test"]
  y_train = loaded_data["y_train"]
  y_test = loaded_data["y_test"]

  X_train = X_train.astype("float") / 256 # 正規化 最大値で割って0~1に収束させる
  X_test = X_test.astype("float") / 256 # 正規化
  y_train = to_categorical(y_train, num_classes) # ono-hot-vector 正解値は1、他は0に変換
  y_test = to_categorical(y_test, num_classes) # ono-hot-vector

  # モデルの定義
  model = model_train(X_train, y_train)
  # モデルの評価(検証)
  model_eval(model, X_test, y_test)


# モデルの定義(畳み込みNN)
def model_train(X, y):
  model = Sequential()  # モデルの作成
  model.add(Conv2D(32,(3,3), padding="same", input_shape=X.shape[1:]))  # 1層目
  model.add(Activation("relu")) # 正のところだけ通して、負の部分を0とする
  model.add(Conv2D(32,(3,3))) # 2層目
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2,2)))  # 一番大きい値を取り出す。
  model.add(Dropout(0.25))  # 25％捨てて、データの偏りを減らす

  model.add(Conv2D(64,(3,3),padding="same"))  # 3層目
  model.add(Activation("relu"))
  model.add(Conv2D(64,(3,3))) # 4層目
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25))

  model.add(Flatten())  # データを一列にする
  model.add(Dense(512)) # 全結合
  model.add(Activation("relu"))
  model.add(Dropout(0.5))
  model.add(Dense(3)) # 最後の出力層のノードは3つ(分類する画像の数)
  model.add(Activation("softmax")) # それぞれの画像と一致している確率を足しと1になると結果を変化する処理

  # 最適化の処理
  opt = tensorflow.keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

  # 損失関数(loss)は正解と推定値がどれぐらい離れているかの数値を小さくする最適化を行う
  model.compile(loss="categorical_crossentropy",
                optimizer=opt,metrics=["accuracy"])
  
  # batch_size 1回のトレーニング(エポック)に使うデータの数
  # nb_epoch トレーニング(エポック)の回数
  model.fit(X,y, batch_size=32, epochs=100)

  # 計算後(fit後)結果をファイルに保存する(モデルの保存)
  model.save("./animal_cnn.h5")

  return model

# 検証関数
def model_eval(model, X, y):
  # verbose 途中経過の表示有無
  scores = model.evaluate(X, y, verbose=1)
  print("Test Loss: ", scores[0])
  print("Test Accuracy: ", scores[1])

# 他のプログラムから参照された時 mainのみ実行する設定
if __name__ == "__main__":
  main()