---
layout: post
title: U combinator, Y combinator和递归
categories: [functional]
tags: [scheme functional]
published: True

date: 2016-01-08
---

最近把Y combinator搞清楚了，总算是迈入了函数式编程的大门，可喜可贺！

## Why Y? Why?
Y combinator有什么用？为什么需要它？我们想象这样一个问题：要在scheme里面实现一个计算阶乘的函数，可能会这样写：

```scheme
(define fact
  (lambda (n)
    (if (= n 0)
        1
        (* n (fact (- n 1))))))

(fact 5)  ; => 120
```

程序没有什么难理解的，使用递归来计算阶乘。但是，如果我们想要在一个匿名函数里实现怎么办？换句话说，怎么在lambda中进行递归？

## U combinator
要在进行递归，我们必须找到一种让函数调用自己的办法。假设我们有以下函数:

```scheme
(define fact1
  (lambda (f)
    (lambda (n)
      (if (= n 0)
          1
          (* n ((f f) (- n 1)))))))

```

不难发现，虽然这个`fact1`函数并不是我们想要的递归函数， 但是`(fact1 fact1) = fact`。原因是我们有
```scheme
	((fact1 fact) 0) = 1
	((fact1 fact1) n) = n * ((fact1 fact1) n-1) ; 当 n != 1
```

这两点。 因此我们可以得到一个递归的阶乘定义：

```scheme
(define fact (fact1 fact1))
```

事实上，这种模式我们可以用在任何递归函数上，我们定义

```scheme
(define (U x) (x x))
(define fact U(fact1))
```

就可以把这种递归的方式进行推广。这里面的U就是传说中的`U combinator`。

## Y combinator
我们刚才提到U combinator可以实现递归，即使编程语言不支持递归，只支持高阶函数。那么Y combinator又是什么呢？
我们考虑以下的函数:

```scheme
(define fact2
  (lambda (f)
    (lambda (n)
      (if (= n 0)
          1
          (* n (f (- n 1)))))))

```

这个函数很像我们的`fact`函数（注意和fact1的区别！），它和`fact`函数之间是什么关系呢？
```scheme
  (fact2 fact) 
= (lambda (n) (if (= n 0) 1 (* n fact (- n 1))))
= fact
```
所以说`(fact2 fact) = fact`， 也就是说， `fact`是`fact2`的 __不动点__。

我们定义一个函数`Y`, 接受一个函数`f`为参数，返回`f`的不动点。这就是`Y combinator`。

```scheme
(Y f) = (f (Y f))
```

所以，假设我们可以得到Y, 那么

```scheme
fact = (Y fact2)
```

通常所说的`Y combinator`是这个：

```scheme
(define Y
  (lambda (f)
    (U(lambda (x) (f (lambda (t) ((x x) t)))))))
```

具体推导过程这里略过，但是我们可以很容易验证满足`(Y f) = f (Y f)`成立。

但实际上不动点函数有很多种，这里介绍另外一种的推导，首先我们根据定义有

```scheme
(define Y
  (lambda (f)
    (f (Y f))
```

事实上这个定义在lazy-scheme里面是可以成立的，但是在applicative order中会无限循环，我们需要这样修改：

```scheme
(define Y
  (lambda (f)
    (f (lambda (t) ((Y f) t)))))
```

这样的定义已经可以work了，可以试试`((Y fact2) 5) = 120`。但是我们还得把函数定义里面的Y给消除掉。怎么消除呢？想起前面提到的`U combinator`了吗？没错，我们就用它:

```scheme
(define Y
  (U
   (lambda (y)
     (lambda (f)
       (f (lambda (t) (((y y) f) t))))))) 
```

用`(y y)`代替需要递归的函数，然后在外围加一个`U`，就得到了一个可以工作的`Y`的定义。这个定义和我们之前介绍的并不一样。其实`Y combinator`不止一个。

## 推荐阅读

[http://mvanier.livejournal.com/2897.html](http://mvanier.livejournal.com/2897.html)

[Fixed-point combinator](https://en.wikipedia.org/wiki/Fixed-point_combinator)

[Lambda calculus](https://en.wikipedia.org/wiki/Lambda_calculus)

[http://matt.might.net/articles/js-church/](http://matt.might.net/articles/js-church/)












