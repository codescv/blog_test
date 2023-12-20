---
layout: post
title: 一个swift编译器范型bug
categories: [iOS]
tags: []
published: True

date: 2015-12-19
---

才写了没几天swift就碰到了一个编译器bug，人品也是杠杠的。我是在写一个core data wrapper的时候碰到这个问题的：

```swift
class Wrapper {
    public func query<T:NSManagedObject>(entity: T.Type) -> QueryOperation<T> {
        let entityName:String = NSStringFromClass(entity).componentsSeparatedByString(".").last!
        return QueryOperation<T>(entityName: entityName, dq: self)
    }
}
```

我本来想这样调用：

```swift
Wrapper(...).query(MyCoreDataModel).filter(condition).all()
```

这样是没问题的。 后来我又写了另一个函数：

```swift
class Wrapper {
    public func insertObject<T:NSManagedObject>(entity: T.Type, context: NSManagedObjectContext) -> T {
        let entityName = NSStringFromClass(entity).componentsSeparatedByString(".").last!
        return NSEntityDescription.insertNewObjectForEntityForName(entityName, inManagedObjectContext: context) as! T
    }
}
```

然后发现像下面这样调用的时候，竟然编译错误：

```swift
Wrapper(...).insertObject(MyCoreDataModel, context: context)
```

编译器显示：

	Expected member name or constructor call after type name

google了一下，正确的用法应该是:

```swift
Wrapper(...).insertObject(MyCoreDataModel.self, context: context)
```

也就是说在swift中，传递类对象必须用类名加上`.self`, 否则会编译错误。而前一个编译成功竟然是bug。不知道是出于什么考虑非要这样, 或者说如果不加的话有什么危害。

详细可以看[这里](http://stackoverflow.com/questions/30209231/swift-ios-sdk-generic-function-with-class-type-name-closure-block-issue)的讨论。



