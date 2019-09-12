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

原始的数据集是以`.gz`保存的，数据的格式是`label, time, duration, id:val, id:val ....，`。这边`label`是指该新闻实际是否被用户点击，`time`和`duration`分别记录了该新闻被点击的时间以及时长，id（uint64的整数）就是对应了`sparse vector`中的位置，也就是如果出现了`id:val`，那么`x[id] = val`。

这边需要注意下，
因为`tf.train.feature`只有三种可选项
1. int64list
2. floatlist
3. bytelist

显然用1，2去存储`uint64`的数据的时候必然会报错，所以这边暂且用3去存储，实际上就是用tf.string去存储这个uint64的数据。
至此，数据就可以全部转换为tfrecord的形式了。

## Sparse Logistic Regression
实现的过程并不复杂，详细代码见`sparse_logistic_regression.py`，和基本的LR区别不大，只需要将原来的`wx+b`用如下语句表示即可
```
z = tf.nn.embedding_lookup_sparse(params=self.w, sp_ids=sparse_ids, sp_weights=sparse_vals, combiner='sum')
predict = tf.nn.bias_add(z, self.b)
```

该代码中给了几个测试用例，直接`python sparse_logistic_regression.py`即可测试。

## Distributed Sparse Logistic Regression
基于Horovod框架，将源代码稍作修改，便可以实现分布式的训练，详细代码见`sparse_logistic_regression_horovod.py`。

## 训练以及测试
`train_sparse_lr.py`和`test_sparse_lr.py`是用来训练和测试sparse logisitc regression的

`train_sparse_lr_horovod.py`和`test_sparse_lr_horovod.py`是用来训练和测试distributed sparse logisitc regression的。
train的时候用法`horovodrun --start-timeout 60 -np 8 -H 100.102.33.44:8 --verbose python train_sparse_lr_horovod.py`
