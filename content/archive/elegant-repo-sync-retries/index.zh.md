---
layout: post
title: '优雅的repo sync自动重试'
date: 2014-03-07 22:15
comments: true
categories: [android]
---
最近在玩Android源代码, 比较头疼的是代码同步实在太痛苦, google的服务器三番两次的被墙, repo 工具又不是很健壮常常会一卡就是几个小时, 必须手动kill了再重试.

于是想写个脚本自动检测, 如果检测到卡住了, 自动kill了重试.


# 第一个版本
这里有几个问题要解决:

1 怎么确定repo sync卡住了?

这个地方我想到的方法是直接用ifstat检测网速, 正常情况下网络流量是有一定值的(例如100k~300k左右的下载速度), 而如果repo sync卡住了, 就会网络流量很低.

这样理论上不是很准, 因为系统上可能有其他程序在跑, 不过就实际情况下来说也够用了.

2 怎么kill ?

首先想到的办法是直接ps |grep [r]epo 然后全部kill掉. 有些不完美: 如果系统上同时有好几个repo程序在跑就一起kill掉了. 容易误伤.

虽然有些缺陷, 我们还是写出一个基本可以工作的版本:

```bash
#!/bin/bash

# FIXME: 只允许同时一个repo运行

kill_prog() {
    # 用ps找出所有的repo, 然后kill掉
    PID=`ps aux |grep python|grep [r]epo |awk '{print $2}'`
    [[ -n $PID ]] && kill $PID
}

start_sync() {
    repo sync &
}

restart_sync() {
    kill_prog
    start_sync
}

# 如果网络流量在retry_delay时间内小于min_speed, 则认为repo sync已经卡住了

min_speed="50"
retry_delay=600
((counter=0))
((n_retries=0))

restart_sync


while [[ 1 ]]; do
    # 用ifstat检测网速
    speed=`ifstat 1 1 | tail -n 1 | awk '{print $1}'`
    
    result=$(echo "$speed < $min_speed" | bc)
    if [[ $result == "1" ]]; then
        ((counter++))
    else
        ((counter=0))
    fi
    if ((counter > retry_delay)); then
        ((counter=0))
        echo "netspeed low. restart!"
        ((n_retries++))
        restart_sync
    fi
done

echo "completed with $n_retries retries"
```

在上面这个版本中, 我们使用ifstat来检测网速, 没秒检测一次, 如果持续一段时间没流量, 就kill掉所有repo进程. 这里有个问题: 如果repo sync正常退出了, 那么也就没有流量了, 所以这个脚本会无限执行, 永远不会停止. 同时, 我们需要知道repo退出时, 是成功的退出, 还是出错退出? 要检测返回值就必须用wait, 然而wait会使主进程 block住, 主进程就无法再去检测了. 为了解决这个问题, 可以使用bash coproc

## Bash Coproc
简单的说bash coproc就是创建一个子进程, 并可以取到子进程的一些信息, 可以和子进程通信. 这里我们需要的是子进程的pid信息, 只要有了这个, 我们就可以在不用wait的情况下知道子进程是否还活着, 如果已经退出了, 我们再用wait去取它的返回值就可以了.  创建coproc的语法如下:
```bash
coproc <name> command  
```
其中<name>为你给他起的名字(可选), command为子进程运行的命令. 运行完后, 可以在$name_PID中取到子进程的PID. 
# 第二个版本
有了coproc, 我们可以写出第二个版本:
```bash
#!/bin/bash

# 当前 repo sync 进程的 pid
PID=

kill_prog() {
    # kill 当前repo sync子进程
    echo "kill : $PID"
    [[ -n $PID ]] && kill $PID
}

start_sync() {
    # 启动子进程(使用coproc)
    coproc syncproc { repo sync; }
    PID=$syncproc_PID
}

restart_sync() {
    kill_prog
    start_sync
}

# 如果网络流量在retry_delay时间内小于min_speed, 则认为repo sync已经卡住了

min_speed="50"
retry_delay=300

((counter=0))
((n_retries=0))

restart_sync

while [[ 1 ]]; do
    # 用ifstat检测网速
    speed=`ifstat 1 1 | tail -n 1 | awk '{print $1}'`
    result=$(echo "$speed < $min_speed" | bc)
    if [[ $result == "1" ]]; then
        ((counter++))
    else
        ((counter=0))
    fi
    if [[ `ps -p $PID| wc -l` == "1" ]]; then
        # 检测到子进程已经退出(ps已经查不到它了)
        
        # 用wait取得子进程返回值
        wait $PID
        
        if [[ $? -eq 0 ]]; then
            echo "sync successful"
            break
        else
            echo "sync failed"
            ((counter=0))
            ((n_retries++))
            restart_sync
            continue
        fi
    fi
    if ((counter > retry_delay)); then
        ((counter=0))
        echo "netspeed low. restart!"
        ((n_retries++))
        restart_sync
    fi
done

echo "completed with $n_retries retries"
```
在这个版本里, 我们知道子进程的pid, 可以有针对性的kill, 并且在子进程正常退出时, 也能检测到, 并及时退出程序.
好了, 现在终于可以自动sync代码了, 下篇就开始写关于android的blog:-)
