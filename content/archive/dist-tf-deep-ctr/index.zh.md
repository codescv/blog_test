---

layout: post

title: "分布式Tensorflow在Deep CTR模型上的实践"

date: "2019-01-18 14:00:00 +0800"

---



最近终于把deep模型推上线， 距离之前训练出sparse LR模型并上线已经过去了半年的时间。是时候总结一下了。



(之前的分布式LR模型的文章在[这里](/dist-tf-for-sparse-models.html))



## deep模型的结构

deep ctr模型基本单元都差不多: linear, embedding, FM, FC, 基本就是这些东西的组合。例如WDL其实可以看作Linear+DNN, deepFM其实就是WDL里把Linear换成FM, sparse特征embedding后直接接FC, 或者embedding交叉后接FC, dense特征直接concat后FC, 等等。



在工程上主要需要考虑的地方是，模型总的大小是多少，每次数据传输有多少，计算量的瓶颈在什么地方。在ctr模型里，一般embedding肯定是最大的，但传输的时候，只需要传batch里涉及到的embedding. dense部分则每个batch训练时都要全部传输，所以也不能太大。另外dense部分的计算复杂度远高于embedding, 因为embedding只是查询和concat, 主要时间是花在网络传输上。



## 训练样本的分配 - revisit



在之前那篇文章里，我使用了一个zookeeper来进行文件分配和状态管理。事实上在阿里巴巴最近开源的x-deeplearning里也是这么实现的。不过，使用zookeeper本身也有其它问题，

假如zookeeper不稳定的话，可能导致训练出问题，有时候文件列表很长造成zookeeper超时，也会给训练造成不必要的麻烦。基于这一点，我设计了另一种解决方案，不需要zookeeper,

也可以进行文件分配，并且更简单:



每个worker拿到完整的文件列表，然后对文件名hash取模来决定自己要处理哪些文件。这里有一个坑，在python 3.x中，对于不同的虚拟机实例，`hash(s)`的值是会变化的，要取得固定

的hash，需要用md5之类的。



## 优雅停止 - revisit



不使用zookeeper的前提下如何让PS感知到worker的停止呢？有两种方案：一种是PS训练完不停止(使用官方的join方法)，直接用一个定时任务去查k8s上master的状态，只要master停止就直接删除任务。另一种方案是把状态写到`model_dir`里，由于`model_dir`是一个共享文件夹, 在nfs或hdfs上，所以所有worker都能读到, 也是一种共享状态的方法。需要注意的坑

是worker可能挂掉重启，所以文件里记录的状态信息需要考虑这一点。



## 使用step, 而不是epoch



在分布式训练中，最好使用step来控制训练的结束，而不是epoch. 为什么呢？因为每个worker速度不一样，如果限定epoch, 它也只是每个worker自己来算，所以速度取决于最慢的那个worker。假如某个worker出问题，或者重启了，那训练可能会很慢甚至永远结束不了。但是step是global的，即使某个worker出问题，只要global step达限，训练就能结束。虽然这样

一来样本不能完全保证每条都跑一样多遍，但是稳定性和速度都有很大提高。



## 模型导出



在LR的模型里，上线我没有用tf-serving, 主要因为LR模型很大，并且非0项又很少，用dense存储很不经济。但是deep模型会比LR模型小很多。这是因为LR里有大量交叉特征，这些特征在

deep模型里被处理成了embedding, 所以大大减小了。例如LR里两个特征交叉，权重的大小数量级在 $$(A\cdot B) $$ , 而如果是用embedding, 则大小为 $$(A+B) \cdot k$$ (k为embedding的维数), 这两者会有很大的差距。至于deep部分的参数，跟embedding比起来非常小，可以忽略不计。 最终我们线上deep模型大小一般在10-20G左右, 所以使用tf-serving是完全可行的。



假如使用Estimator, 那么模型导出已经实现好了。如果不用的话，则需要在做模型导出的时候，创建两张图，一张图用于训练，输入为TFRecord, 输出为optimizer; 另一张图用于导出模型，输入为一个placeholder, 输出为prediction, 两张图需要共享模型变量. 例如这样:



```python

# model function with shared variables

def model_fn(features):

    with tf.variable_scope(reuse=tf.AUTO_REUSE):

        y_pred = whatever_net(features)

    return y_pred



# training graph

features, label = dataset.make_one_shot_iterator().get_next()

y_pred = model_fn(features)

loss = make_loss(y_pred, label)

train_op = optimizer.minimize(loss)



# prediction graph

example = tf.placeholder(dtype=tf.string)

feature_ = parse_example(example)

y_ = model_fn(feature_)



while training:

    session.run(train_op)



# export model

tf.saved_model.simple_save(

    session,

    inputs={'example': example},

    outputs={'pred': y_}

)



```



## 测试模型



模型导出后，可以借助command line工具测试一下导出后的模型。使用方法在[https://www.tensorflow.org/guide/saved_model#cli_to_inspect_and_execute_savedmodel](https://www.tensorflow.org/guide/saved_model#cli_to_inspect_and_execute_savedmodel)



## train/prediction一致性



一般来说使用tf-serving不太容易碰到线上线下不一致的问题。不过我们还是很幸运的遇到了。常见的问题有:



1. 训练时有dropout, 导出时的图要记得去掉。

2. 训练时的batch norm, 导出时要把moving average设置为不更新。



## 性能



在LR模型中，由于计算量小，worker的CPU使用普遍很低。在deep模型里CPU使用率较高，一般集群中CPU能基本用满，而内存有较大富裕。

对于大量embedding的模型来说，GPU性价比仍然不高，所以我们也没有用GPU来进行训练。



网络带宽仍然是一大瓶颈，多个任务同时训练时，cpu很容易scale而网络却不容易scale。所以推荐样本使用GZIP压缩。



训练时如果遇到性能问题，可以先查查各个变量的大小，是否因为传输数据量不合理导致。这个是最常见的问题。推荐使用[这里](/debug-dist-tf-ps.html)的方法.





## 样本处理



样本处理中也有一些坑。在tensorflow里，缺失的embedding会变成0向量，无法训练，所以最好在样本处理时就给一个特殊的值，以免出问题。dense特征也需要额外的处理。



## 参数调整



deep模型里一些参数也会比较影响效果。我这边的经验是: optimizer > learning rate > embedding size > batch size > regularization.

其中dropout基本不需要，用了以后效果极差。因为样本只跑一遍很难overfit, 所以l2 reg基本也不需要。



Tensorflow的Adam optimizer在分布式上性能很差，应该是个bug, 不知道现在修了没有；据说有个LazyAdam, 我试了一下效果不行。所以最后用的是Adagrad.

Learning rate对效果影响还是比较可观的，值得多调调。batch size对效果和训练速度都有影响，也可以调一下。有人反应同步

训练的效果会比异步训练效果好很多，我之前在LR上也发现了这一点，但是同步的稳定性和速度都要差很多，所以最后还是用了异步, 最终线上效果比LR涨了5%-10%的样子。