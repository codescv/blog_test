---

layout: post

title: Redis Highlights (2) Redis中的事件处理

categories: [Redis]

tags: [Redis, Databases]

published: True



---



# Redis中的事件处理



和MySQL不同，Redis是一个*单线程*数据库。所有的读写操作都在主线程里完成。(当然，会有一些操作，例如rdb文件的写操作可以发生在后台线程里) 主线程的循环可以在`ae.c`里看到大致的流程：



{% highlight c %}

void aeMain(aeEventLoop *eventLoop) {

    eventLoop->stop = 0;

    while (!eventLoop->stop) {

        if (eventLoop->beforesleep != NULL)

            eventLoop->beforesleep(eventLoop);

        aeProcessEvents(eventLoop, AE_ALL_EVENTS);

    }

}

{% endhighlight %}



其中`aeProcessEvents`就是处理所有事件的函数。Redis中主要两类事件，一是File Events, 就是和客户端网络交互中发生的事件，一是Time Events，目前用来运行`serverCron`.



`aeProcessEvents`里关键代码：



{% highlight c %}

shortest = aeSearchNearestTimer(eventLoop);

tvp = ...

...

numevents = aeApiPoll(eventLoop, tvp);

// 处理file events

...

...

// 处理time events

processTimeEvents(eventLoop)

{% endhighlight %}



所以消息循环的逻辑如下：首先用aeApiPoll获取当前pending的file events，然后处理file events. `aeApiPoll`的第二个参数表示block的时间，它是根据最近一个timer决定的。假如最近一个timer要在1秒后触发，那么poll的timeout就是1秒。这样做的目的是减少忙等待。



file events处理完后就会处理time events. 这里只有一个timer就是`serverCron`，刷新rdb, 刷新各种计时器，以及一些housekeeping的任务都放在这里完成。
























