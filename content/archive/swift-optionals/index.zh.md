---

layout: post

title: swift optionals

categories: [iOS]

tags: [iOS swift]

published: True



---



swift中一个objective-c没有的概念就是optionals. optionals在作用上类似于objc中的nil pointer, 但更加安全。



# Optional的基础用法

{% highlight swift %}

class Animal {

    func name() -> String { return "animal" }

}



var a1: Animal?

a1 = Animal()

a1!.name()   // animal

var a2: Animal?

a2?.name()   // nil

a2!.name()   // runtime error



var a3: Animal = nil // compiler error

var a3: Animal = a2  // compiler error

{% endhighlight %}

optional有两种状态，一种是nil, 一种是某个对象的值.



使用在类型名后加`?`的方式来创建一个optional. 



optional可以像赋予对象，也可以赋予nil。



普通变量不可以赋nil, 也不可以赋optional。(均为 compiler error)



当需要取optional里面对象的值时，使用`?`或者`!`，该过程称为`unwrap`，会把optional变为普通变量. `?`类似于objc中的nil pointer，当optional变量为nil时进行no-op, 不会报错; 而`!`当变量为nil时会crash，类似于其他语言中的NPE.



有了optional后，我们就可以方便的使用类似objc中nil pointer的特性，但是同时又可以用crash来及时的找出程序中的bug，可以说是兼具两者的优点。





# implicit unwrap (隐式解包)

{% highlight swift %}

var a2: Animal?

a2?.name()   // nil



var a3: Animal! = a2 // a3 = nil

a3.name()  // crash



var a4: Animal = a2! // crash

{% endhighlight %}

当我们希望把一个optional解开时, 可以用在类名后加`!`的方式。这样得到的变量a3类型*仍为*optional，只是在调用其方法时会自动用`!`来解开，不需要再显式解包。注意a3和a4的不同：a3是一个optional，而a4是一个普通变量，所以在调用a2!的时候会直接crash。



隐式解包在和objc代码接口时常常会用到，因为objc里的类变量对于swift看来都是optional。



# optional binding

使用`if let`可以把check optional和unwrap一起进行，如果不为nil则调用if内部，同时将optional转化为实际类型:



{% highlight swift %}

if let x = someOptional {

	// someOptional is not nil, x gets the value

}

{% endhighlight %}



# optional chaining

可以使用?将多个optional调用连接，任何一步optional为nil都为noop。例如：



{% highlight swift %}

x.prop1?.prop2?.method()

{% endhighlight %}



# 方法+`?`

objc中使用`respondsToSelector:`来判断一个对象是否能调用某个方法。在swift中，可以直接用`someObj.method?()`来完成同样的功能，如果method不存在就是no-op.



# 构造函数+`?`

init?() 可以用来定义一个可失败的构造函数。当构造失败时返回nil。



# 类型转换

swift在进行类型转换时，使用`as`关键字。向上转换（转换为父类）使用`as`，编译器可以自动判断其安全性；向下转换时使用`as?`会生成optional, 转换不成功为nil；使用`as!`会自动unwrap optional，如果转换失败会crash。



# Optional的定义

swift 中的Optional 定义如下(简化了很多细节)：



{% highlight swift %}

public enum Optional<Wrapped> {

    case None

    case Some(Wrapped)

}

{% endhighlight %}



可以看到，Optional其实是个enum, 所以以下代码是等效的：



{% highlight swift %}

var a1: Animal?

a1 = Animal()



if let a = a1 {

    print(a.name())

}



// pattern matching

if case .Some(let a) = a1 {

    print(a.name())

}



// pattern matching using switch

switch(a1) {

case .None:

    print("nil")

case .Some(let a):

    print("animal \(a.name())")

}

{% endhighlight %}

所以，`?`, `!`等，都是一些语法糖，都可以转换为普通的pattern matching代码。






















