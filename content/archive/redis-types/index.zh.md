---
layout: post
title: Redis Highlights (1) Redis中的类型和数据结构
categories: [Redis]
tags: [Redis, Databases]
published: True

date: 2016-12-11
---

# Redis中的类型和数据结构

redis对外有5中基本类型，分别是string `t_string.c`, list `t_list.c`, hash `t_hash.c`, set `t_set.c` 和 zset (ordered set) `t_zset.c`.
这5种类型是“接口”而不是“实现”，因此redis得以根据不同的情形自由选择不同数据结构的实现，这也是redis在设计上的高明之处。

5种基本类型对应了int `object.c`, embstr `object.c`, raw `sds.c`, linkedlist `adlist.c`, ziplist `ziplist.c`, skiplist `t_zset.c`, ht `dict.c`, intset `intset.c` 这8种数据结构的实现。

类型与数据结构实现的对应关系如图。

[![redis types](redis_types.png "redis types")](https://www.planttext.com/?text=VLB13SCW3Fmp1GjqWQgFV3i63Mv10H9XggXHXwyfHC5fbH_dEv_F5Xqc5TFJEo6IJGva16rHfjS4A5NG4a8_QXiUA8GD2U9TzI0nHIer1MSnDT0eDAqSMdB9KFHE8KggrXVu6xbM4DLSNsRbdGq3wE-SKbZir20oohO5u50vKOBAo_kanpcSikmihtvou84wVlYIub12sHWlygIDhh6OX2ksJRXFFjgn3rUONJ_jpnObp1Tf-NtpmjX_mxbYFJ5t4Hq4JP_f0m00)

实用`type KEYNAME`可以查看某个key对应的类型，而`object encoding KEYNAME`可以查看该key内部的实现。

## string
string 有三种实现方式，分别是`int`, `embstr`和`raw`.
长度比较短的整数会使用`int`实现。长度比较短的字符串会使用`embstr`, 更长的会使用`raw`。
`embstr`和`raw`的区别在于，`embstr`吧`redisObject`和`sds`放在一块连续空间里面，这样申请内存和释放内存都只需要一次调用。带来的坏处是，`embstr`是只读的，如果调用`append`等操作则自动升级为`raw`。

对于`int`实现的string，如果调用`strlen`和`gettrange`等会产生额外开销。如果需要强制使用`raw`来实现, 可以用`setrange`。

## zset
zset在元素较少时，使用`intset`实现。
在元素较多时，使用`skiplist`和`dict`一起实现。其中`skiplist`用于提供顺序相关的操作，而`dict`用于快速查询score.

关于skiplist: 这是一种用多级链表加快查询速度的数据结构。Insert, delete, search的复杂度均为o(logn), 还可以进行高效的range query, 功能与性能与红黑树，B树相当，但实现更简单。CLRS里有详细的解释。










