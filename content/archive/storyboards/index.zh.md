---
layout: post
title: '[iOS] Storyboard 技巧总结'
categories: [iOS]
tags: [iOS]
published: True

date: 2015-11-14
---

以前做iOS开发的时候主要还是用nib，最近几年Stroyboards变得比较流行。这几天集中学习了Storyboard的相关内容，总结下经验。

# 阻止segue
有时候在点击一个按钮的时候希望进行一个check，满足条件才继续进行。在iOS6以后可以做到这一点：
```objc
- (BOOL)shouldPerformSegueWithIdentifier:(NSString *)identifier sender:(id)sender
```
注意如果在代码中出发segue是不会自动调用这个函数的，需要你自己来调用。

# unwind segue
unwind segue用于"回退"一个segue。这个回退不一定是完全的go back。
应用场景：

- 注册新用户向导，下一步，下一步，然后点击重置回到第一个Controller
- 在一个tab中进行了操作，希望把当前的操作做完后切换到另一个tab的另一个controller.


# 使用多个Stroyboard
Storyboard的一个重要缺点，也是我一直很嫌弃它也没怎么用它的原因在于，它把所有的ViewController放在一个文件里，这样有3个重大缺陷：

1. 对重用很不利
2. 对多人合作时的版本控制merge很不友好。
3. ViewController多了以后，想找到相关的东西很费事。

解决办法是： 使用多个Storyboard。

## Xcode 7之前
将Stoyboard拆成多个文件，例如每个tab一个storyboard。然后使用`[UIStroyboard instantiateInitialViewController]`和`[UIStroyboard instantiateViewControllerWithIdentifier:]` 来加载每个storyboard中的controller, 并把这些controller 设置为根tabcontroller的content controller.

当需要跨storyboard进行segue时，可以创建一个custom segue， 然后改写`perform`方法，将需要跳转到的scene encode到segue identifier里。可以看[这篇文章](http://spin.atomicobject.com/2014/03/06/multiple-ios-storyboards/)和[代码](https://github.com/qblu/LinkedStoryboardSegue)。

## Xcode 7之后
Xcode 7增加了一个new feature, 可以直接框选一组scene， 然后在菜单中选`Editor -> refactor to storyboard` 来创建新的storyboard。refactor之后，segue扔保持其连接。

同时，也可以手动从库中拖出storyboard reference 来直接创建跨storyboard的segue，非常方便。




