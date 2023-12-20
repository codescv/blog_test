---
layout: post
title: ReactiveCocoa 代码阅读笔记 (1) 双向绑定
categories: [iOS]
tags: [iOS ReactiveCocoa]
published: True

date: 2016-01-30
---

我们首先从一个双向绑定的例子开始。假设我们的`UIViewController`里有一个`UITextField`和一个`UILabel`， 我们希望在`UITextField`输入时，
`UILabel`里面同步显示一样的内容。我们通过把`UITextField`和`UILabel`绑定到同一个`model`，也就是`UIViewController`的`name`属性上
来实现这一点。

```objc
RAC(self.nameTextfield, text) = RACObserve(self, name);
RAC(self.nameLabel, text) = RACObserve(self, name);
    
[self.nameTextfield.rac_textSignal subscribeNext:^(id x) {
    self.name = x;
}];
// this has the same effect as the above
// RAC(self, name) = self.nameTextfield.rac_textSignal;
```

当`model`变化时，`self.nameTextfield`和`self.nameLabel.text`都会随之变化。而当`UITextField`中的输入发生变化时，`model`也会发生变化。 `UITextField`和`model`发生了双向绑定，而`label`和`model`是单向绑定。

我们把上面的代码用`clang -E`展开，看看到底发生了什么：

```objc
[[RACSubscriptingAssignmentTrampoline alloc] initWithTarget:(self.nameTextfield) nilValue:(((void *)0))][@(((void)(__objc_no && ((void)self.nameTextfield.text, __objc_no)), "text"))] = ({ __attribute__((objc_ownership(weak))) id target_ = (self); [target_ rac_valuesForKeyPath:@(((void)(__objc_no && ((void)self.name, __objc_no)), "name")) observer:self]; });
```

简化一下:
```objc
[[RACSubscriptingAssignmentTrampoline alloc] initWithTarget:(self.nameTextfield) nilValue:0][@"text"] = ({ __weak id target_ = (self); [target_ rac_valuesForKeyPath:@("name") observer:self]; });
```

等等，`@(((void)(__objc_no && ((void)self.nameTextfield.text, __objc_no)), "text"))` 这个是什么东西？ 这个东西其实就等效于`@"text"`，但是为什么要前面这一长串呢？这是为了在编译期防止你引用了错误的属性，比如你写了`@RAC(self.nameTextField, somethingNotexist)`，编译器就会给出提示说这个属性不存在。clever.


## RAC

我们来看前半部分，也就是RAC的展开:

```objc
[[RACSubscriptingAssignmentTrampoline alloc] initWithTarget:(self.nameTextfield) nilValue:0][@"text"] = ...
```

RAC宏第一个参数是target，也就是需要绑定的对象；第二个参数是keyPath, 也就是对象中需要绑定的属性名。RAC实际上是创建了一个RACSubscriptingAssignmentTrampoline对象，并调用其`setObject:forKeyedSubscript:`方法。

```objc
// RACSubscriptingAssignmentTrampoline.m
- (void)setObject:(RACSignal *)signal forKeyedSubscript:(NSString *)keyPath {
	[signal setKeyPath:keyPath onObject:self.target nilValue:self.nilValue];
}

// RACSignal+Operations.m
- (RACDisposable *)setKeyPath:(NSString *)keyPath onObject:(NSObject *)object nilValue:(id)nilValue {
    // ...
	RACDisposable *subscriptionDisposable = [self subscribeNext:^(id x) {
		// ...
		__strong NSObject *object __attribute__((objc_precise_lifetime)) = (__bridge __strong id)objectPtr;
		[object setValue:x ?: nilValue forKeyPath:keyPath];
	} error:^(NSError *error) {
		// ...

		[disposable dispose];
	} completed:^{
		[disposable dispose];
	}];
	// ...
}
```

可见RAC的下标set操作对右边的信号调用了subscribeNext, 并在所有的next event中通过`object setValue:forKeyPath`修改属性的值。在这个例子里，也就相当于`[self.nameTextField setValue:x forKeyPath: @"text"]`


## RACObserve
RACObserve的展开:

```objc 
({ __weak id target_ = (self); [target_ rac_valuesForKeyPath:@("name") observer:self]; });
```

rac_valuesForKeyPath:observer:调用了另一个方法:

```objc 
// NSObject+RACPropertySubscribing.m
- (RACSignal *)rac_valuesAndChangesForKeyPath:(NSString *)keyPath options:(NSKeyValueObservingOptions)options observer:(__weak NSObject *)weakObserver {
	// ...
	return [[[RACSignal
		createSignal:^ RACDisposable * (id<RACSubscriber> subscriber) {
			//...

			return [self rac_observeKeyPath:keyPath options:options observer:observer block:^(id value, NSDictionary *change, BOOL causedByDealloc, BOOL affectedOnlyLastComponent) {
				[subscriber sendNext:RACTuplePack(value, change)];
			}];
		}]
		takeUntil:deallocSignal]
		setNameWithFormat:@"%@ -rac_valueAndChangesForKeyPath: %@ options: %lu observer: %@", self.rac_description, keyPath, (unsigned long)options, strongObserver.rac_description];
	// ...
}
``` 

这个方法创建了一个RACSignal, 这个Signal在keypath发生变化时会发出“通知”.
这个Signal由RACKVOTrampoline实现，RACKVOTrampoline封装了KVO，将在keypath发生变化时发送信号。

```objc 
// NSObject+RACKVOWrapper.m
- (RACDisposable *)rac_observeKeyPath:(NSString *)keyPath options:(NSKeyValueObservingOptions)options observer:(__weak NSObject *)weakObserver block:(void (^)(id, NSDictionary *, BOOL, BOOL))block {
	// ...
	RACKVOTrampoline *trampoline = [[RACKVOTrampoline alloc] initWithTarget:self observer:strongObserver keyPath:keyPathHead options:trampolineOptions block:^(id trampolineTarget, id trampolineObserver, NSDictionary *change) {
		// ... 当KVO发生后进入这里，这里会调用前面传入的block
	}

}
``` 


## rac_textSignal

知道了`RAC`和`RACObserve`，接下来我们来看`rac_textSignal`的实现：

```objc
// UITextField+RACSignalSupport.m
- (RACSignal *)rac_textSignal {
	@weakify(self);
	return [[[[[RACSignal
		defer:^{
			@strongify(self);
			return [RACSignal return:self];
		}]
		concat:[self rac_signalForControlEvents:UIControlEventEditingChanged | UIControlEventEditingDidBegin]]
		map:^(UITextField *x) {
			return x.text;
		}]
		takeUntil:self.rac_willDeallocSignal]
		setNameWithFormat:@"%@ -rac_textSignal", self.rac_description];
}
```

可见rac_textSignal监控UIControlEventEditingChanged \| UIControlEventEditingDidBegin事件，subscriber将获得textfield.text

## weakify 和strongify
RAC中常见的`@weakify` 和 `@strongify`展开如下：

```objc
@weakify
@autoreleasepool {} __attribute__((objc_gc(weak))) __typeof__(self) self_weak_ = (self);;

@strongify
__typeof__(self) self = self_weak_;
```

为了看起来美观，用了一个`autoreleasepool`的hack，使得宏的前面可以用`@`，有点意思。


## textView
前面`Textfield`的`rac_textSignal`使用监控control events的方式来获取变化。但是`UITextView`并没有这些event。所以RAC使用了delegate的方式来实现`UITextview`的`rac_textSignal`。但是又有一个问题：如果想同时使用delegate怎么办呢？

### delegate forwardInvocation
RACDelegateProxy会用forwardInvocation的方式发送给self.delegate
所以如果需要同时使用，先设置delegate,再使用rac_textSignal即可。

```objc
// UITextView+RACSignalSupport.m
- (RACSignal *)rac_textSignal {
	@weakify(self);
	RACSignal *signal = [[[[[RACSignal
		defer:^{
			@strongify(self);
			return [RACSignal return:RACTuplePack(self)];
		}]
		concat:[self.rac_delegateProxy signalForSelector:@selector(textViewDidChange:)]]
		reduceEach:^(UITextView *x) {
			return x.text;
		}]
		takeUntil:self.rac_willDeallocSignal]
		setNameWithFormat:@"%@ -rac_textSignal", self.rac_description];

	RACUseDelegateProxy(self);

	return signal;
}
```

`signalForSelector`是RAC中一个重要的方法，它的作用简单来说就是勾住一个方法，当这个方法被调用时把所有的参数放在一个tuple里，当做信号发出来。因此，`RACUseDelegateProxy(self)`把`UITextView`的`delegate`指向`RACDelegateProxy`，而`RACDelegateProxy signalForSelector:@selector(textViewDidChange:)`就可以捕捉所有原本调用`delegate`的方法，并发出`textView`实例作为信号，并在`reduceEach`中获得其`text`属性。

# 参考

关于ReactiveCocoa:

[http://cocoasamurai.blogspot.jp/2013/03/basic-mvvm-with-reactivecocoa.html](http://cocoasamurai.blogspot.jp/2013/03/basic-mvvm-with-reactivecocoa.html)

[http://www.raywenderlich.com/62699/reactivecocoa-tutorial-pt1](http://www.raywenderlich.com/62699/reactivecocoa-tutorial-pt1)
