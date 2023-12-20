---
layout: post
title: Java中的Variance
categories: [Java]
tags: [Java, OOP]
published: True

date: 2016-12-19
---

# Java中的Variance

默认情况下，Java中的Generics都是Invariant. 例如ArrayList<Integer>不是ArrayList<Number>的子类。如下代码说明了原因：
```Java
ArrayList<Number> x = new ArrayList<Integer>(); // 假如能编译通过
x.add(1);
x.add(3.5); // 错误！
int y = x.get(0); // 可以
```

这是因为，我们对一个Generic Collection的操作，往里面放东西和从里面往外拿东西所需要的类型是不一样的。假如Array<P>是Array<C>的父类，根据里氏替换原则，Array<P>的对象一定可以被Array<C>的对象替换。

假如我们需要往里放东西，我们已知 Array<P>::add(P p), 根据替换原则 Array<C>::add应该能接受P, 而我们已知Array<C>::add能接受C, 所以C >= P. 

假如我们要往外取东西, P p = Array<P>::get, 根据替换原则 Array<C>.get()应该能被P接受, 所以C能被P接受， 则 P >= C.

综合上面两条，对于一个Generic来说，假如Array<P>是Array<C>的父类，则P = C. 这就是Invariant.

同时我们可以知道两点结论：

如果P是C的父类，假如这个Generic，我们只需要往外取东西(函数的返回值)，那么Class<P> 就是 Class<C> 的父类，这就是covariant. 典型的例子: `Iterator<? extends T>`.

如果P是C的父类，假如这个Generic，我们只需要往里放东西(函数的参数)，那么Class<C> 就是 Class<P> 的父类，这就是contravariant. 典型的例子: `Consumer<? super T>`.

参考实例代码体会。在这里, List<? extends Number> 可以接受 List<Double>, 这就是covariant. Consumer<? super Double> 可以接受 Consumer<Number>, 这就是 contravariant.

```Java
public class Main {
    private static List<Double> fList = Arrays.asList(1.0,2.0,3.0);

    public static void main(String[] args) {
        System.out.println(intSum(fList));

        forEach(new Consumer<Number>() {
            @Override
            public void accept(Number number) {
                System.out.println("number: " + number);
            }
        });
    }

    public static Number intSum(List<? extends Number> l) {
        int sum = 0;
        for (Number n : l) {
            sum += n.intValue();
        }
        return sum;
    }

    public static void forEach(Consumer<? super Double> consumer) {
        for (double x : fList)
            consumer.accept(x);
    }
}
```












