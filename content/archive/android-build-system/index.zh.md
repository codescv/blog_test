---
layout: post
title: 'Android 源代码上手指南 - Build System初探'
date: 2014-03-21
comments: true
categories: [android]
---
Android总体来说是个挺好用的系统, 虽然有不少这样那样让我觉得不喜欢的地方. 不过好在它是开源的, 于是很多不爽的地方可以自己改. 研究Android源代码也有一段时间了, 在github上开了个坑, 把自己的一些想法加了进去, 取名为yacm:

[http://github.com/codescv/yacm](http://github.com/codescv/yacm)

之所以叫yacm主要是因为它是基于[CyanogenMod](http://cyanogenmod.com)改的 (yet another cyanogenmod). 里面加了一些我觉得很需要但是CM没有提供的功能, 例如:

- 去掉了拨VPN必须设置锁屏密码的限制. (Android脑残点之一)
- 将媒体扫描改为白名单模式, 只有白名单内的几个文件夹(如DCIM, Pictures等)会被扫描, 再也不用担心相册里出来一大堆乱七八糟的广告图片什么的了. (Android脑残点之二)
- 拨号盘增加首字拼音搜索. (中国用户必备)
- 增加了一些方便开发者的工具, 例如下拉快捷菜单中的ADB开关, 电源按钮开关等

在研究的过程中也走了不少弯路, 这里尽量把这些东西都记下来, 一方面提醒自己, 一方面也希望对后来人有所帮助.

首先, 研究android代码应该从哪里开始? 我首先是找到了几本国内的android源代码分析的书, 一般套路是先讲讲怎么编译, 然后就直接开始研究代码了. 其实这种方式很不好, 因为android工程极其庞大(我测试了一下, 光makefile就有几千个), 无论你怎么分析源代码也只是九牛一毛, 完全抓不住主线. 即使把这些书都看完了, 也还是停留在纸上谈兵的阶段.

后来我找到一本书叫做 [Embedded Android](http://shop.oreilly.com/product/0636920021094.do), 这本书对我启发非常大, 主要是它真的很懂刚上手Android源代码的程序员需要什么. 首先大概的介绍了一下android的架构, 然后就开始将构建系统(Build System), 然后讲各种android specific的知识点, 例如硬件, shell命令, init.rc, system server等等, 几乎很少涉及代码. 所谓构建系统就是讲android中这么多的工程是如何build到一起的, 哪些工程最后会进入最终的系统包. 事实上, 对于很多程序员来说, android中最陌生的是构建系统, 因为里面用了非常多的古怪语法(你一开始都不敢相信这些是Makefile), 而Java或者C代码很多情况下不需要解释和分析都能看懂. 只有先了解了构建系统, 才能有方向, 否则以上来就研究代码必然是像无头苍蝇乱撞, 似懂非懂的改了一通代码, 结果发现这段代码根本没编译进最后的项目. 

所以说, 我的这个blog系列也准备先抛开代码不谈, 先介绍构建系统, 然后做一些小的修改, 例如预装一些软件, 修改一些默认设置等等, 先玩转构建系统, 搞清来龙去脉, 之后会发现hack代码是水到渠成的事.

本文假定你已经从CM上sync了最新的代码, 并且能够编译成功了. 由于我的手机是HTC One, 所以我的代码是基于HTC One来改的, 不过其他手机原理也都是一样的. 如果你还没有编译成功, [这里](http://wiki.cyanogenmod.org/w/Build_for_m7)有个传送门.

当你需要编译android源代码的时候, 其实有三步, 缺一不可:

```bash
source build/envsetup.sh
lunch cm_m7-userdebug
mka bacon
```

第一步是加载了 build/envsetup.sh中的各种函数 (后面会讲到), 第二步主要是检验makefile的合理性(在一定程度上), 以及设置和具体device相关的各种环境变量, 第三步是真正的build. 接下来我们一一分析.

# source build/envsetup.sh

这一步是运行了`build/envsetup.sh`这个文件. 在我的机器上, 运行输出是这样的:

```bash
including device/generic/armv7-a-neon/vendorsetup.sh
including device/generic/goldfish/vendorsetup.sh
including device/generic/mips/vendorsetup.sh
including device/generic/x86/vendorsetup.sh
including vendor/cm/vendorsetup.sh
including sdk/bash_completion/adb.bash
including vendor/cm/bash_completion/git.bash
including vendor/cm/bash_completion/repo.bash
```

说明这个文件会加载一些其他的文件, 比如很多目录下的`vendorsetup.sh`. 最后三个是用来做命令行补全的, 例如`adb rem<TAB>` 会给你补全成`adb remount`. 先不管.

我们看看envsetup.sh的内容, 这个文件总的来说是一堆util函数/命令的集合. 其中有几个特别有用的:


    懒人必备型
    croot: 切换到顶部. android目录很深, 经常进去就回不去了, 于是这个可以一键回家.
    mm: 单独编译某个工程.
    godir: 匹配一个目录名称, 进入该目录. (例如godir m7会匹配到device/htc/m7, 虽然也会有很多其他目录)

    关键命令
    lunch - 后面要用的, 用来选择设备
    mka - 约等于一个智能多进程的make

刨去各种函数的定义, 这个脚本的副作用很少, 只有在文件底部有这么几行:

```bash
for f in `/bin/ls vendor/*/vendorsetup.sh vendor/*/*/vendorsetup.sh device/*/*/vendorsetup.sh 2> /dev/null`
do
    echo "including $f"
    . $f
done
unset f
```

这其实是加载了vendor和device下的vendorsetup.sh脚本, 也就解释了之前的输出.

这个文件有很多函数(2000行), 我们从哪里入手呢? 我们按时间顺序, 先看看各个被include的vendorsetup.sh文件做了什么吧:

```bash
for combo in $(curl -s https://raw.github.com/CyanogenMod/hudson/master/cm-build-targets | sed -e 's/#.*$//' | grep cm-11.0 | awk {'print $1'})
do
    add_lunch_combo $combo
done
```

其实各个子文件夹下的`vendorsetup.sh`的内容都差不多, 都是在调用`add_lunch_combo`, 只是后面的参数不一样. 我们从名字就可以猜出, `add_lunch_combo`其实是在增加`lunch`的品种--也就是增加一种设备选项而已. 而cm下的这个文件, 事实上是从网络上一个文件里读到了所有cm官方支持的设备品种, 然后把它们都加上, 这样就可以在`lunch`的时候选择了.

我们看一下`add_lunch_combo`的定义, 其实很简单:
```bash
unset LUNCH_MENU_CHOICES
function add_lunch_combo()
{
    local new_combo=$1
    local c
    for c in ${LUNCH_MENU_CHOICES[@]} ; do 
        if [ "$new_combo" = "$c" ] ; then 
            return
        fi
    done 
    LUNCH_MENU_CHOICES=(${LUNCH_MENU_CHOICES[@]} $new_combo)
}
```
只是把`new_combo`加到一个全局变量LUNCH_MENU_CHOICES里面而已.

# lunch
目前为止我们并没有做什么事情, 只是定义了很多函数, 然后添加了一些combo. 而且我们知道接下来就只要看`lunch`这一个函数就可以了. 这个函数做了些什么呢?

```bash
function lunch() {
	# ...
  # 前面只是验证参数合法性, 略
  
  # selection也就是运行lunch带的参数, 如果selection = cm_m7-userdebug, 则 product = cm_m7
  local product=$(echo -n $selection | sed -e "s/-.*$//")
 
 	# 验证product是否合法
  check_product $product
	
  # roomservice (见后面分析)
  if [ $? -ne 0 ]
  then
        # if we can't find a product, try to grab it off the CM github
				# 一看到上面这句注释就知道不是重点, 略
   else
   			build/tools/roomservice.py $product true
   fi
   
   # variant = userdebug
   local variant=$(echo -n $selection | sed -e "s/^[^\-]*-//")
   
   # 验证variant的合法性
   check_variant $variant

   #...
	 # 这三个全局变量很重要, 后面到处都在用!
   export TARGET_PRODUCT=$product
   export TARGET_BUILD_VARIANT=$variant
   export TARGET_BUILD_TYPE=release

   # 不要在意这种细节
	 fixup_common_out_dir
   
   # 见后面分析
   set_stuff_for_environment
   
   # 一看就知道没啥用
   printconfig
}
```

这里罗嗦几句:

roomservice是用来获取特定设备的dependencies的, 例如htc手机需要htc的kernel, 它的代码从哪里获取呢? 就是roomservice管理, 依据cm.dependencies文件获取到repo地址, 然后放到`.repo/local_manifest`里面. 噢对了, 如果你还不熟悉repo, 请[出门右转](http://source.android.com/source/using-repo.html)

`set_stuff_for_environment` 里面设置了一些shell prompt, path什么的. 如果你跟我一样设置了自己的shell提示符, 那么最好设置一个`STAY_OFF_MY_LAWN`环境变量,这样android就不会乱搞shell prompt了.

另外还有个值得一提的函数是`get_build_var`, 它在很多地方都用到, 例如`check_product`等等. 它的作用是获取一个build variable的值.

```bash
function get_build_var()
{
    # ...
    CALLED_FROM_SETUP=true BUILD_SYSTEM=build/core \
      make --no-print-directory -C "$T" -f build/core/config.mk dumpvar-$1
}
```

可以看到`get_build_var`其实是执行了 `build/core/config.mk` 中的 `dumpvar-$1` 目标(替换成实际的环境变量, 例如`dump-TARGET_PRODUCT`). 关于makefile的语法规则我们在后面会更深入讨论. 

综上所述, `lunch`命令其实是设置了很多的环境变量, 而这些环境变量在后面的make会被用到. 想看看到底设置了多少环境变量吗? 

    set > /tmp/result

打开`/tmp/result`文件看看, 可以看到所有被设置过的环境变量值, 这些值在以后make的阶段都有很重要的用途.

唔, 现在已经很晚了, 那么今天就先讲到这里吧, 后面就是重头戏`mka bacon`了, 不过在这之前可能要先介绍下make的各种语法. 各位晚安~!


































