# Distributed Sparse Logistic Regression

## 数据预处理

Feature是一个[1,2^64]Sparse Vector构成。这个feature包含了用户的信息，新闻的信息以及用户和新闻的一些交互信息。之所以有如此大的维度的feature，是因为有一些连续的feature value被量化了。因此编码完之后导致了维度很大。

首先`toTFRecords.py`把数据集转换为TFRecords的形式，这样读取的时候可以调用tf的接口，将数据pipeline的方式读取。\
使用方法：
```
mkdir tfrecords
cd tfrecords
python toTFRecords.py --input raw_data_directory
```

原始的数据集是以`.gz`保存的，数据的格式是`label, time, duration, id:val, id:val ....，`。这边label是指该新闻实际是否被用户点击，time和duration分别记录了该新闻被点击的时间以及时长，id（uint64的整数）就是对应了sparse vector中的位置，也就是如果出现了id:val，那么x[id] = val。
