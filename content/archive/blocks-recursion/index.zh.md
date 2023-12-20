---
layout: post
title: '[iOS] Blocks 中的递归'
date: 2015-11-08 21:00
comments: true
categories: ios
---

# 一点废话
已经有一年多没有写blog了。最近一年人变的好懒，我觉得这样很不好，希望从现在开始可以坚持再写一写blog。
最近又重新在看了很多ios的东西，觉得ios开发还是很有意思的，想自己动手做app，也许哪天能回归也说不定。

# 回归正题
今天正好看到blocks，突然想到一个问题：在blocks里面要如何实现递归？我先按自己的想法试了下，用blocks来实现一个递归版本的阶乘函数`fact`：

```objc
int (^fact)(int) = ^int(int a) {
    if (a == 0) return 1;
    else return a * fact(a-1);
};

NSLog(@"fact(3)=%d", fact(3));
```

运行以上代码后会crash，报错EXC_BAD_ACCESS，为何？原因是因为`非__block`变量会在block被创建的时候就直接捕获，而这个时候`fact = xxx` 这个赋值还没有发生，因此block中的`fact`变量会永久成为nil。

知道原因后，我们可以把`fact`声名为`__block`变量，这样该变量会被block放到堆上并进行retain操作，从而保持住值。下面来看修改后的版本：

```objc
__block int (^fact)(int) = ^int(int a) {
    if (a == 0) return 1;
    else return a * fact(a-1);
};
    
NSLog(@"fact(3)=%d", fact(3));
```

改成`__block`后程序可以正确输出`fact(3) = 6`，但是我们还有一个问题：循环引用。由于fact和block之间互相强引用，会造成内存泄漏。关于这一点，我们可以通过代码来验证：

```objc
__weak int (^var)(int);
{
    __block int (^fact)(int) = ^int(int a) {
        if (a == 0) return 1;
        else return a * fact(a-1);
    };
    
    var = fact;
    
    NSLog(@"fact(3)=%d", fact(3));
}
NSLog(@"var: %@", var);
```

可以看到，我们声明了一个weak变量，如果里面没有内存泄漏的话，花括号结束后`fact`离开作用域，应该释放其内存，从而使`var`变为nil。而这里我们发现`var`在花括号结束后还有值，说明内存没有被正确的释放掉。

假设我们在`return 1`这里释放内存行不行呢？例如下面：

```objc
__weak int (^var)(int);
{
    __block int (^fact)(int) = ^int(int a) {
        if (a == 0) { fact = nil; return 1; }
        else return a * fact(a-1);
    };
    
    var = fact;
    
    NSLog(@"fact(3)=%d", fact(3));
}
NSLog(@"var: %@", var);
```

看上去是可以了，然而连续调用两次`fact`就会crash，因为这时候`fact`已经被设置为nil了。

因此我们需要一个weak变量来指向block，来帮助我们释放内存。最后的解决方案如下：

```objc
__weak int (^var)(int);
{
    int (^fact)(int);
    __block __weak int (^weakFact)(int);
    weakFact = fact = ^int(int a) {
        if (a == 0) {
            return 1;
        }
        else return a * weakFact(a-1);
    };
    
    var = fact;
    
    NSLog(@"fact(3)=%d", fact(3));
}
NSLog(@"var: %@", var);
```

我们使用两个变量，一强一弱，用弱指针进行递归，强指针使得block不会提前被释放。当强指针离开作用域时，block被正确释放。

可见，要在Objective-C中正确的使用block来递归还是有很多坑的。这一点也是由于其引用计数的策略造成的。