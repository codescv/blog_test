---

layout: post

title: 'Bash在cd进入目录时自动启动脚本'

date: 2014-03-08 16:35

comments: true

categories: 

---

玩android代码进入目录后总是需要`source build/envsetup.sh`, 比较烦, 能不能在cd进入时自动完成这个功能呢?



可以的, 首先我们在`.bashrc`中重新定义cd这个函数:



{% highlight bash %}

mycd() {  

    \cd $@  

    local hook_file=.cd_hook  

    if [[ -f $hook_file ]]; then  

        . $hook_file  

    fi  

}   

alias cd='mycd'  

{% endhighlight %}



 这样, 在进入一个目录后就会检测该目录下是否有`.cd_hook`文件, 如果有的话就自动`source`之.

然后我们在源代码目录下新建一个`.cd_hook`文件:



{% highlight bash %}

type lunch >/dev/null 2>&1  # 用检测lunch函数是否有的方式确定是不是已经source过了  

if [[ $? -ne 0 ]]; then  

    source build/envsetup.sh  

fi  

{% endhighlight %}



注意, 这样修改cd是有一定安全性问题的, 因为恶意程序可以在目录下放置`.cd_hook`文件来做任何事情. 因此最好还是不要`alias cd`. 在实际使用的时候, 我用的是`alias c='mycd'`, 这样我只有用"c"命令进入目录时会运行`.cd_hook`.



顺便提一句, 在脚本中, 为了防止alias, 可以用`\`转义. 例如`\ls ~`会确保调用的不是alias后的ls.
