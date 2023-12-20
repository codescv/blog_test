---
layout: post
title: Python MetaProgramming
categories: [Python]
tags: [Python, MetaProgramming]
published: True

date: 2016-11-08
---

# class decorators
像函数一样，class也可以有decorator. 例如:

```python
@debugattr
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def debugattr(cls):
    '''
    Decorator that adds logging to attribute access
    '''
    orig_getattribute = cls.__getattribute__

    def __getattribute__(self, name):
        print('Get:', name)
        return orig_getattribute(self, name)

    cls.__getattribute__ = __getattribute__
    return cls
```

class的decorator也是类似的语法糖，`Point = debugattr(Point)`.
在这个例子里面，我们"打开"了`Point`这个类，并修改了它的`__getattribute__`方法的定义。

# metaclass
metaclass是一种class，它创建的instance是class. metaclass几乎总是会继承自type.

一般来说，创建一个metaclass需要改写`__new__`方法，来实现自定义的功能:

```python
class debugmeta(type):
    '''
    Metaclass that applies debugging to methods
    '''
    def __new__(cls, clsname, bases, clsdict):
        clsobj = super().__new__(cls, clsname, bases, clsdict)
        clsobj = debugmethods(clsobj)
        return clsobj

def debugmethods(cls):
    '''
    Apply a decorator to all callable methods of a class
    '''
    for name, val in vars(cls).items():
        if callable(val):
            setattr(cls, name, debug(val))
    return cls
```

在上面的例子里展示了如何将class decorator应用在metaclass里。多数情况下，能用metaclass实现的功能用decorator也能实现，但是metaclass的好处
在于会自动继承, 也就是说parent class的metaclass会自动成为subclass的metaclass. 例如：

```python
class Base(metaclass=meta):
    pass

class A(Base):  # type(A) = meta && A's metaclass = meta
    pass
```

# decorator/class decorator/metaclass 之比较

总结一下，在python中，实现wrapping的方式有三种：

- decorator: wrapping functions
- class decorators: wrapping classes
- metaclasses: wrapping class hierarchies

另一个区别：decorator作用于decorated object定义之后，而metaclass作用于之前。

# descriptor
descsriptors用来控制`.`运算符的语义。

```python
class Descriptor:
    def __init__(self, name=None):
        self.name = name

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        raise AttributeError("Can't delete")

class Stock(Structure):
    shares = Descriptor("aa")
```

# super
super() 会调用下一个`mro`里对象中的方法(而不是父类)。

# import hooks
`sys.meta_path`中包含了一个Finder list，可以用来影响module的加载. 例如，以下代码
定义了一个StructImporter，可以从xml中读取数据，并创建一个python module.

```python
class StructImporter:
    def __init__(self, path):
        self._path = path

    def find_module(self, fullname, path=None):
        name = fullname.rpartition('.')[-1]
        if path is None:
            path = self._path
        for dn in path:
            filename = os.path.join(dn, name+'.xml')
            if os.path.exists(filename):
                return StructXMLLoader(filename)
        return None

import imp
class StructXMLLoader:
    def __init__(self, filename):
        self._filename = filename
        
    def load_module(self, fullname):
        mod = sys.modules.setdefault(fullname,
                                     imp.new_module(fullname))
        mod.__file__ = self._filename
        mod.__loader__ = self
        code = _xml_to_code(self._filename)
        exec(code, mod.__dict__, mod.__dict__)
        return mod

import sys
def install_importer(path=sys.path):
    sys.meta_path.append(StructImporter(path))
```

# 参考
[Python 3 Metaprogramming](https://www.youtube.com/watch?v=sPiWg5jSoZI&t=590s)