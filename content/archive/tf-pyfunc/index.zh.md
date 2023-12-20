---
layout: post
title: "在Tensorflow中使用python来解析样本"
date: "2019-01-03 14:00:00 +0800"
---

Tensorflow中一般会使用[TFRecord](https://www.tensorflow.org/guide/datasets)作为样本格式，这样可以有feature column, Estimator等比较方便的支持。
但有时TFRecord不能满足需求，或者只是想做一些quick and dirty的实验，这时可能用python解析样本会更方便。但是如果用feed dict + python的话，就用不了Dataset API了。
此时可以用py_func这个API来将任意python函数转换为tf operation.

例如，我们的样本是如下的格式:
```
1.0,2,33,252
0.7,1,34,323,112
```
其中第一个值为label, 后面不定长的整数列表为feature. 要解析这样一个样本，我们可以写一个python函数:
```python
import tensorflow as tf
import numpy as np

def parse(line):
    label, *features = line.decode('utf-8').split(',')
    return np.float32(label), np.array([np.int64(feature) for feature in features])
```
接下来使用pyfunc来将它和Dataset API结合起来，就可以进行样本的解析了：
```python
ds = tf.data.TextLineDataset('test.txt')
line = ds.make_one_shot_iterator().get_next()
label, feature = tf.py_func(parse, [line], [tf.float32, tf.int64])

try:
    with tf.Session() as sess:
        while True:
            print(sess.run([label, feature]))
except tf.errors.OutOfRangeError:
    pass
```
如果用了Dataset的batch方法，parse中的line会变成一个数组，在python中做相应的修改即可。