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
  loaded_data = np.load("./drive/MyDrive/ML/image_div/animal_aug.npz")

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
  # Conv2D(フィルター数, (フィルターサイズ), パディング, ストライド)
  # フィルター数 多いほど、より多くの特徴を抽出できますが、計算コストが増加
  #            16~512以上 32, 64, 128 などの2の累乗の値がよく使われます。
  # フィルターサイズ 小さなフィルターは詳細な特徴を捉えるのに適しており、大きなフィルターは広い範囲の特徴を捉えるのに適しています
  #                1x1, 3x3, 5x5, 7x7, 11x11
  model.add(Conv2D(32,(3,3), padding="same", input_shape=X.shape[1:]))  # 1層目
  # Activation(活性化関数)
  # relu 隠れ層
  model.add(Activation("relu")) # 正のところだけ通して、負の部分を0とする
  model.add(Conv2D(32,(3,3))) # 2層目
  model.add(Activation("relu"))
  # MaxPooling2D(プーリングサイズ)
  # プーリング層で使用されるウィンドウのサイズを指定
  model.add(MaxPooling2D(pool_size=(2,2)))  # 一番大きい値を取り出す。
  # Dropout(ドロップアウト率)
  # 過学習を防ぐために、ネットワーク内の一部のユニットをランダムに無効化する割合を指定
  # 高いほど、過学習のリスクが減り、学習が進みにくくなる
  # ドロップアウト率は0から1の範囲
  # 0.0 ～ 0.7
  model.add(Dropout(0.25))  # 25％捨てて、データの偏りを減らす

  model.add(Conv2D(64,(3,3),padding="same"))  # 3層目
  model.add(Activation("relu"))
  model.add(Conv2D(64,(3,3))) # 4層目
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25))

  model.add(Flatten())  # データを一列にする
  # Dense(ニューロン数)
  # ユニット数が多いと、より複雑な関係を学習できるが、計算コストが増え、過学習のリスクも高まる
  # 32 ～ 1024以上, 64, 128, 256, 512, 1024
  model.add(Dense(512)) # 全結合
  model.add(Activation("relu"))
  model.add(Dropout(0.5))
  model.add(Dense(3)) # 最後の出力層のノードは3つ(分類する画像の数)
  # Activation(活性化関数)
  # softmax 出力層
  model.add(Activation("softmax")) # それぞれの画像と一致している確率を足しと1になると結果を変化する処理

  # 最適化の処理
  # Optimizer(学習率, 減衰率)
  # 学習率 小さすぎると遅い、大きすぎると最適な解に収束しない(精度が下がる)
  #       0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001
  # 減衰率 学習率をエポックごとに減少させるためのパラメータ
  opt = tensorflow.keras.optimizers.RMSprop(learning_rate=0.001, decay=1e-6)

  # 損失関数(loss)は正解と推定値がどれぐらい離れているかの数値を小さくする最適化を行う
  model.compile(loss="categorical_crossentropy",
                optimizer=opt,metrics=["accuracy"])
  
  # batch_size 1回のトレーニング(エポック)に使うデータの数
  #            大きいと学習が効率的だがメモリの使用量が増える
  #            16 ～ 512, 32, 64, 128
  # epochs トレーニング(エポック)の回数
  #        多いほど多く学習するが、過学習のリスクがある
  #        10 ～ 500以上, 50, 100, 200
  model.fit(X,y, batch_size=32, epochs=100)

  # 計算後(fit後)結果をファイルに保存する(モデルの保存)
  model.save("./animal_cnn_aug.h5")

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