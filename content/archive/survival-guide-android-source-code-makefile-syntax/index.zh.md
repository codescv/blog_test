---

layout: post

title: 'Android 源代码上手指南 - Makefile语法'

date: 2014-03-22 23:05

comments: true

categories: 

---

上一篇我们将到android build system, 讲了envsetup.sh和lunch, 在继续研究build过程之前有必要先讲讲Makefile的语法. android使用`GNUMake 3.81`(**不支持其他版本!**), 因此有许多GNU的特定语法, 和一般的Makefile长得很不一样. (为了叙述方便, 本文中凡是提到`Makefile`的地方, 均指`GNU Makefile`.) Makefile的基本语法是这样的:



    TARGET: <DEPENDS>

    <TAB>   instructions



运行`make TARGET`命令时, 根据TARGET对应的DEPENDS来确定依赖关系, 根据修改时间来判断某个TARGET是否需要rebuild.如果需要, 则执行`TARGET`对应的instructions. 这就是Makefile的基本原理.



# .PHONY targets

在Makefile中, 有时会遇到标明为`.PHONY`的target, 它表示这个target只是一个名字, 不是一个文件. 例如:

{% highlight makefile %}

clean:

		rm *.o temp

{% endhighlight %}

这个名为`clean`的target指示makefile去做一些删除临时文件的指令, 但如果你的目录下刚好有个名为`clean`的文件, 又因为clean没有任何的dependencies, make会认为`clean`这个文件是最新的, 不需要执行. 这时我们就要告诉make, `clean`这个target只是一个名称而已, 并不是一个文件:



{% highlight makefile %}

.PHONY: clean

{% endhighlight %}



# multiple targets

当一行写有个多个target时, 相当于把这些target分开写, 并使用一样的commands. 例如:

{% highlight makefile %}

foo bar:

		echo $@    			

{% endhighlight %}

等价于:

{% highlight makefile %}

foo:

		echo $@

bar:

		echo $@

{% endhighlight %}

其中$@会被替换为target名称.



在`GNUMake`中, 有一些特定的语法在android中广泛使用, 下面一一介绍一下:



# define

在`Makefile`中, 定义一个变量可以用`=`或者`:=`.

其中`=`和`:=`的区别在于, `=`是`recusively expanded variable`, 它在每次被用到的时候都会被求值; 而`:=`是`simply expanded variable`, 它在定义的时候就被展开. 例如:

{% highlight makefile %}

x = foo 

var1 = $(x)

var2 := $(x)

x = bar 

target:

        echo var1: $(var1)

        echo var2: $(var2)

{% endhighlight %}



执行`make`会显示:



    echo var1: bar

    var1: bar

    echo var2: foo

    var2: foo



除了`=`和`:=`, 还有一个`?=`, 它只在变量没有被定义时起作用, 常用于设置默认值.



define 也可以用来定义一个变量, 但define更强的一点在于可以定义多行变量. 例如:



{% highlight makefile %}

define command

@echo foo

@echo bar

endef



target:

			$(command)

{% endhighlight %}

输出:



  foo

  bar



其中`@`放在一个命令前, 表示不输出这一行执行的命令.

不过, 在Makefile中有一个`call`函数, 它可以"调用"一个变量, 因此变量可以用来实现类似于宏展开的功能. 例如:



{% highlight makefile %}

define concat

$1,$2

endef



target:

			@echo $(call concat,foo,bar)

{% endhighlight %}



会输出:



    foo,bar



这里有个坑: `$(call concat,foo,bar)`注意我没有输入空格, 如果我写成`$(call concat, foo, bar)`, 那么在concat展开的时候, 参数`$1`就会等于前面带一个空格的`_foo`, `$2`就会等于前面带一个空格的`_bar`(我这里用下划线来表示空格作为警示). 这个在用的时候要格外小心!



# include

`include`用于包含另一个Makefile, 这个在比较大规模的项目中比较有用, 有助于Makefile的重用和模块化. 例如你可以将一些常用的宏扩展和变量定义放在一个`definitions.mk`中, 然后从你的Makefile中include之. 



{% highlight makefile %}

include some_file.mk

-include other_file.mk

{% endhighlight %}



当include一个文件而这个文件不存在时, Make会报错, 而在`include`之前加上`-`表示文件不存在时不要报错. 这个常用来包含可选的文件.



# if

在make语法中有一些常用的条件语句, 类似于C语言中的`#ifdef`和`#if`等. 例如:

{% highlight makefile %}

ifeq($(var),foo)

	echo foo

else

	echo not foo

endif

{% endhighlight %}

`ifeq`检测后面两个值是否相等. 类似的还有`ifneq`, `ifdef`, `ifndef`检测一个变量是否被定义, 等等.



# 函数

makefile中内置了一些函数, 方便实现一些常用的功能. 调用一个函数的语法是:

{% highlight makefile %}

$(function arguments) 或者

${function arguments}

{% endhighlight %}



下面列举一些常见函数:



## 字符串操作

- $(subst _from_, _to_, _text_)  字符串替换

- $(strip _string_) 去掉开头和末尾的空白. (还记得前面说的那个坑吗?)

- $(filter _pattern_..., _text_) 取出所有匹配的文本, 丢到所有不匹配的. 例如`$(filter %.c %.s,$(sources))`取出`sources`变量中所有以`.c`和`.s`结尾的文本. 

- $(filter-out _pattern_..., _text) 和上面相反, 取出所有不匹配的.

- $(word _n_, _text_) 取出`text`中第`n`项.



## 文件名操作

- $(dir _names_...) 取出`names`中所有文件的目录部分.

- $(notdir _names_...) 和上面相反, 取出非目录部分(纯文件名部分).

- $(add_prefix _prefix_, _names_...) 给文件加前缀

- $(wildcard _pattern_) 获取匹配通配符的文件名. 例如`objects := $(wildcard *.o)`返回所有`.o`结尾的文件.



## foreach

$(foreach _var_, _list_, _text_)  将`var`作为形式参数, `list`作为列表, 扩展`text`的内容. 相当于python中的:

{% highlight python %}

	for var in list:

      text

{% endhighlight %}



## call

在前面提到了, 用于"展开"一个宏(变量)



## eval

`eval`是个很重要的函数, 可以用来实现Makefile的模板. 比较关键的一点是需要知道传给`eval`的参数被展开了两次, 首先是传给eval之前, 然后再被当作makefile的语句include到当前的地方. 这个我们之后会继续讲到, 先放一个例子在这里.

{% highlight makefile %}

PROGRAMS    = server client



server_OBJS = server.o server_priv.o server_access.o

server_LIBS = priv protocol



client_OBJS = client.o client_api.o client_mem.o

client_LIBS = protocol



# Everything after this is generic



.PHONY: all

all: $(PROGRAMS)



define PROGRAM_template =

 $(1): $$($(1)_OBJS) $$($(1)_LIBS:%=-l%)

 ALL_OBJS   += $$($(1)_OBJS)

endef



$(foreach prog,$(PROGRAMS),$(eval $(call PROGRAM_template,$(prog))))



$(PROGRAMS):

        $(LINK.o) $^ $(LDLIBS) -o $@



clean:

        rm -f $(ALL_OBJS) $(PROGRAMS)



{% endhighlight %}



## 输出调试

这些语句被用于输出调试信息: (debug和研究时都非常有用!)

- $(warning _text_...)

- $(info _text_...)

- $(origin _variable_) 可以告诉你一个变量是从哪里来的, 是环境中定义的, 还是makefile中定义的, 还是命令行中定义的. 这个在调试的时候也很有用. 参考[这里](http://www.gnu.org/software/make/manual/make.html#Functions)



## 运行shell语句

- $(shell _command_) 用于运行一个shell语句, 然后将输出返回. 这个也非常有用, 例如有了这个你就可以用你自己喜欢的工具例如perl之类来进行字符串操作了, 可以不用费力学习和记忆makefile中相应的函数的语法.




















