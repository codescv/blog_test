---
layout: post
title: "Recursive Genenerators in Python"
date: "2018-08-14 20:00:00 +0800"
---

假设想写一个In order traversal, 在python里是这样:

```python
result = []
def inorder(p):
    if p is None:
        return

    inorder(p.left)
    result.append(p.val)
    inorder(p.right)
```

如果想用generator实现这个算法应该怎么做呢?

```python
def inorder(p):
    if p is None:
        return

    inorder(p.left)
    yield p.val
    inorder(p.right)

for v in inorder(tree):
    print(v)
```

会发现只能打印出root节点。问题出在哪里呢？因为yield在递归调用中并不会起作用。在第一次调用inorder的时候，yield出了p.val, 但是里面的inorder调用并不会yield出值来。要想让递归的调用里也能往外yield, 需要显式指明这一点:

```python
def inorder(p):
    if p is None:
        return

    yield from inorder(p.left)
    yield p.val
    yield from inorder(p.right)
```


