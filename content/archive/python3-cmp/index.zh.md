---

layout: post

title: Python3中排序的cmp函数的替代方法

categories: [Python]

tags: [Python]

published: True



---



在python3中的`sort`函数取消了对`cmp`函数的支持。在大多数场景下，使用`key`确实是更好的方法，但是如果我们有逻辑不方便用`key`表示呢？

比如以下的例子:



{% highlight python %}

def my_cmp(x, y):

    if x is None and y is not None:

        return -1

    if x is not None and y is None:

        return 1

    else:

        if x > y:

            return 1

        elif x < y:

            return -1

        else:

            return 0



# python2

sorted([1,None,5,2], cmp=my_cmp)

{% endhighlight %}



像这样复杂的逻辑不太容易转成`key`函数，或者我们不想改变已有使用`cmp`的代码，该怎么办呢？我们可以自己写一个函数，把`cmp`函数转换成`key`函数。



{% highlight python %}

class KeyWrapper(object):

    def __init__(self, cmp, val):

        self.val = val

        self.cmp = cmp



    def __lt__(self, other):

        return self.cmp(self.val, other.val) == -1





def cmp_to_key(func):

    def key(x):

        return KeyWrapper(func, x)

    return key



print(sorted([1,3,None,2], key=cmp_to_key(my_cmp)))

{% endhighlight %}



我们在key函数中返回一个`KeyWrapper`对象，这个对象保存了`cmp`函数和实际要比较的值的引用，然后改写`KeyWrapper`的`__lt__`比较函数，使用`cmp`函数

进行大小比较即可。



事实上`python3`已经为我们想到了这一点，我们只需要用`functools.cmp_to_key`函数就可以了。


