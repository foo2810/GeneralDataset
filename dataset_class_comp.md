# 機械学習ライブラリのデータセット周りの調査

## 1. 対象ライブラリ

+ Keras
+ Chainer
+ Pytorch
+ scikit-learn
+ (Theano)
+ (Caffe)

---

## 2. 各ライブラリのデータセットの形式

### **Keras**

**kerasにおける学習データの扱い**

- 入力データのNumpy配列と対応するラベルデータのNumpy配列をそれぞれfitに渡す
- kerasの形式に従ったジェネレータを作成してfit_generatorに投げる
- Sequenceクラスを継承したデータセットクラスのようなものを作成しfit_generatorに投げる

### **Chainer**

**Chainerにおける学習データの扱い**

- chainer.datasetクラスを継承して作成したデータセットクラスを使って学習

    データセットクラス内でミニバッチも作成している模様

### **Pytorch**

**Pytorchにおける学習データの扱い**

ぶっちゃけPytorchのデータセットの形はなんでもいい(自分でバッチ作ってforwardしてbackwardしてstepすればいいから)

- torchvisionを用いたデータセットクラスを作成し利用
- 入力データとラベルデータをそれぞれ何らかの配列で用意し利用

### **scikit-learn**

基本的に入力データとラベルデータのNumpy配列をそれぞれ用意してfitで学習

ただし，irisデータセットなどのsklearnで用意されている有名データセットはBunchというdictionary-likeなクラスのインスタンスとして提供されている模様
ただ結局このデータセットからNumpy配列に変換するのでBunchはsklearnの学習用のデータセットとは言えないかもしれない．

Bunchは今回の汎用データセットの枠組みを作る上でその実装方法が参考になりそう．

### **Theano**

sklearnと同じ形式だと思われる．

### **Caffe**

sklearnと同じ形式だと思われる．

---

## **3. 実装計画**

別資料を参考


