---

layout: post

title: "使用tfdbg Debug Tensorflow代码"

date: "2018-10-14 14:00:00 +0800"

---



由于Tensorflow使用静态图，先build graph然后run graph, 所以用一般的python debugger不能单步跟踪计算过程。如果想要debug

计算的中间过程, 一种简单的办法是在fetches中加入想要查看的tensor, 但是在调试中如果需要查看很多个tensor的值，这种方法不是很方便。

Tensorflow官方提供了tfdbg这个命令行工具来解决这个问题。



# 准备工作

## 使用tfdbg来包装Session

对一般使用Session的Tensorflow程序，可以使用`LocalCLIDebugWrapperSession`来启用tfdbg，例如



```python

from tensorflow.python import debug as tf_debug

sess = tf_debug.LocalCLIDebugWrapperSession(sess)

```



## 使用hook来注入Estimator

如果使用Estimator, 那么需要使用`LocalCLIDebugHook`来启用tfdbg, 例如



```python

from tensorflow.python import debug as tf_debug



# Create a LocalCLIDebugHook and use it as a monitor when calling fit().

hooks = [tf_debug.LocalCLIDebugHook()]

```



# tfdbg的使用

以[debug_mnist.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/debug/examples/debug_mnist.py)这个脚本为例。

首先运行脚本:

```bash

python debug_mnist.py --debug

```

进入tfdbg的主界面:



![tfdbg-main](tfdbg-main.png)



上面一行`run #1: 1 fetch (accuracy/accuracy/Mean:0); 2 feeds`表示当前这次`Session.run`的信息, 对应到代码里:



```python

135   for i in range(FLAGS.max_steps):

136     acc = sess.run(accuracy, feed_dict=feed_dict(False))

137     print("Accuracy at step %d: %s" % (i, acc))

```



在run_info里可以看到fetch是`accuracy/accuracy/Mean:0`, feed有两个:`x-input:0`和`y-input:0`.



## 使用run命令运行一个完整的step

```

tfdbg> run

```

使用`run`命令可以进行一次`Session.run`. 执行后的结果如图:



![tfdbg-run](tfdbg-run.png)

在这个界面可以看到运行这个step中所有的Operation, tensor的大小，以及运行时间。

点击其中的Tensor或者运行命令`pt <tensor_name>`,可以看见某个Tensor的值。例如点击`Softmax:0`以后，出现如下界面:



![tfdbg-printtensor](tfdbg-printtensor.png)



点击`node_info`查看该节点的输入输出，以及在代码的什么位置被定义.



点击`list_inputs`和`list_outputs`可以查看输入输出的依赖树.



在`print_tensor`界面可以看到这个Softmax的函数输出的形状是`(10000, 10)`, 因为这是一个test batch, batch size是10000.



使用`pf`命令可以打印feed, 从而验证这一点:

```

tfdbg> pf input/x-input:0

```



![tfdbg-pf](tfdbg-pf.png)



## 导出Tensor到文件

当Tensor比较大的时候，如果希望把Tensor导出进行进一步分析, 例如我们想导出`hidden/weights/Variables:0`, 可以用如下命令:



```

tfdbg> eval -a '`hidden/weights/Variable:0`' -w '/tmp/variable.npy'

或者

tfdbg> pt -s hidden/weights/Variable:0 -w '/tmp/variable.npy'

```



![tfdbg-dump](tfdbg-dump.png)



之后可以用numpy来读取这个变量, 例如

```python

import numpy as np

var = np.load('/tmp/variable.npy')

```



`eval`命令还可以支持更加复杂的语法, 例如

```

tfdbg> eval "np.matmul((`output/Identity:0` / `Softmax:0`).T, `Softmax:0`)"

```



## 单步跟踪

使用`invoke_stepper`命令进入单步模式:



![tfdbg-stepper](tfdbg-stepper.png)



接下来使用`s`命令就可以运行一个step(注意，这里step和之前step的概念不同)

使用`s -t <num>` 可以运行`num`个step.

使用`exit`运行余下的step并退出单步模式。



## filter(类似条件"断点")

默认情况下打印的tensor有点多，如果希望按照自己设定的条件来打印相关tensor, 可以使用`filter`. 例如，设置如下filter:



```python

def my_filter_callable(datum, tensor):

    return 'Softmax' in datum.tensor_name

sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type=FLAGS.ui_type)

sess.add_tensor_filter('my_filter', my_filter_callable)

```



则只有名称包含`Softmax`的tensor会被打印:



```

run -f my_filter

```



![tfdbg-filter](tfdbg-filter.png)



再比如，想要运行到包含`nan`或者`inf`(一般情况下意味着训练有问题)的tensor可以使用:

```

tfdbg> run -f has_inf_or_nan

```



`has_inf_or_nan`是一个默认被注册的`filter`.