---
layout: post
title: ReactiveCocoa 代码阅读笔记 (2) Signal的实现机制
categories: [iOS]
tags: [iOS ReactiveCocoa]
published: True

date: 2016-02-07
---

上篇讲了RAC和RACObserve两个宏的实现机制，但都是把RACSignal当作一个黑盒来理解。
本篇详细讲解Signal的内部机制。

## 创建Signal

Signal有很多种创建方式，例如`[UITextView rac_textSignal]`是用`signalForSelector:`的方式。这里我们先看看
自定义创建的Signal:

```objc
// RACSignal.m
+ (RACSignal *)createSignal:(RACDisposable * (^)(id<RACSubscriber> subscriber))didSubscribe {
	return [RACDynamicSignal createSignal:didSubscribe];
}

// RACDynamicSignal.m
+ (RACSignal *)createSignal:(RACDisposable * (^)(id<RACSubscriber> subscriber))didSubscribe {
	RACDynamicSignal *signal = [[self alloc] init];
	signal->_didSubscribe = [didSubscribe copy];
	return [signal setNameWithFormat:@"+createSignal:"];
}
```

`createSignal`方法接收一个block, 把这个block存在`_didSubscribe`属性中。顾名思义，这个block会在`subscribe`的时候被调用。

## subscribe一个signal
下面看最常用的`subscribeNext`的定义

```objc
// RACSignal.m
- (RACDisposable *)subscribeNext:(void (^)(id x))nextBlock error:(void (^)(NSError *error))errorBlock completed:(void (^)(void))completedBlock {
	RACSubscriber *o = [RACSubscriber subscriberWithNext:nextBlock error:errorBlock completed:completedBlock];
	return [self subscribe:o];
}
```

`subscribeNext`会创建一个`RACSubscriber`对象， 它保存了`nextBlock`, `errorBlock`和`completedBlock`，这些block将在`sendNext`, `sendError`和`sendCompleted`时调用:

```objc
// RACSubscriber.m
- (void)sendNext:(id)value {
	@synchronized (self) {
		void (^nextBlock)(id) = [self.next copy];
		if (nextBlock == nil) return;

		nextBlock(value);
	}
}

- (void)sendError:(NSError *)e {
	@synchronized (self) {
		void (^errorBlock)(NSError *) = [self.error copy];
		[self.disposable dispose];

		if (errorBlock == nil) return;
		errorBlock(e);
	}
}

- (void)sendCompleted {
	@synchronized (self) {
		void (^completedBlock)(void) = [self.completed copy];
		[self.disposable dispose];

		if (completedBlock == nil) return;
		completedBlock();
	}
}
```

可以看到, `sendNext`是线程安全的，并且`nextBlock`是和`sendNext`在同一个线程上执行的。

我们回到`RACSignal`, `subscribeNext`会调用`subscribe`:

```objc
// RACDynamicSignal.m
- (RACDisposable *)subscribe:(id<RACSubscriber>)subscriber {
	NSCParameterAssert(subscriber != nil);

	RACCompoundDisposable *disposable = [RACCompoundDisposable compoundDisposable];
	subscriber = [[RACPassthroughSubscriber alloc] initWithSubscriber:subscriber signal:self disposable:disposable];

	if (self.didSubscribe != NULL) {
		RACDisposable *schedulingDisposable = [RACScheduler.subscriptionScheduler schedule:^{
			RACDisposable *innerDisposable = self.didSubscribe(subscriber);
			[disposable addDisposable:innerDisposable];
		}];

		[disposable addDisposable:schedulingDisposable];
	}
	
	return disposable;
}
```

`RACPassthroughSubscriber`可以看作和`subscriber`功能一样，这里不细讲，所以`subscribe:`就是调用了`RACScheduler.subscriptionScheduler schedule:` 并传入一个block, 在这个block中调用了`didSubscribe`，也就是`createSignal`中的block。所以我们可以得到一个结论：对于`createSignal`创建的信号，每`subscribe`一次，其block就被调用一次。

`RACScheduler.subscriptionScheduler`返回一个单例的`RACSubscriptionScheduler`对象, 其`schedule`方法:

```objc
// RACSubscriptionScheduler.m
- (RACDisposable *)schedule:(void (^)(void))block {
	NSCParameterAssert(block != NULL);

	if (RACScheduler.currentScheduler == nil) return [self.backgroundScheduler schedule:block];

	block();
	return nil;
}
```

这个方法会找`RACScheduler.currentScheduler`, 这个方法在主线程上或者在`schedule`内部(schedule中的schedule)会返回非`nil`, 此时直接调用block; 如果不在主线程上，最外层schedule会使用`backgroundScheduler`，也就是在一个background queue上执行block.

 因此, 如果createSignal在主线程上执行，(之后调用subscribe时)signal的`didSubscribe`(也就是`createSignal`传入的block)也会在主线程上执行。

一般自定义的signal会在`didSubscribe`这个block中(或者其后续)调用`[subscriber sendNext]`等方法，这样`subscribeNext:`中传入的block就会执行。应该注意到的是`subscirbeNext:`中传入的block会和`sendNext`在同一个线程上执行。

# rac_signalForSelector:
`rac_signalForSelector: `这个方法很重要，它为各种`UIView`的delegate方法封装提供了基础。这个方法产生一个signal, 这个signal会在receiver执行某个selector时发送函数调用的参数。以UITextView为例:

```objc
// NSObject+RACSelectorSignal
- (RACSignal *)rac_signalForSelector:(SEL)selector;

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

这里会用一个`self.rac_delegateProxy`作为`TextView`的proxy，调用`[self.rac_delegateProxy signalForSelector:@selector(textViewDidChange:)]` 也就是产生了一个信号， 当`[delegate textViewDidChange:]`调用时触发。

`rac_signalForSelector`中创建了一个`subject`(一个可以手动控制进行`sendNext`的信号), 做了很复杂的swizzling, 其中和发送信号相关的是`RACSwizzleForwardInvocation`, 这个方法改写了原本的`forwardInvocation`. 然后将自己的对应selector的方法替换成`forwardInvocation`，这个`forwardInvocation`里面会调用`[subject sendNext:]`发送信号给subscriber.

一句话概括：`rac_textSignal`会"劫持" `UITextView.delegate的textViewDidChange:`方法，然后在这个方法调用时，把这个方法传入的参数发送给subscriber。这个劫持过程的实现是由`[NSObject rac_signalForSelector:]`方法完成的。
