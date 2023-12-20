---

layout: post

title: "Debug Tensorflow的C++代码"

date: "2018-10-09 11:00:00 +0800"

---



单步调试python代码，直接用IDE就可以了(例如Intellij Idea). 但如果要跟踪C++代码，就要使用GDB(Linux使用GDB, Mac OS上要用LLDB).



# 准备工作

## 编译安装带调试信息的Tensorflow



安装编译工具见: [https://www.tensorflow.org/install/source](https://www.tensorflow.org/install/source)



首先使用`-c dbg`来编译安装tensorflow(以MacOS上编译tensorflow 1.8.0为例):



```bash

bazel build -c dbg //tensorflow/tools/pip_package:build_pip_package || exit 1

rm -rf /tmp/tf_pkg && bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tf_pkg || exit 1

pip uninstall -y tensorflow || exit 1

echo "Build Successful:" $(ls /tmp/tf_pkg)

pip install /tmp/tf_pkg/tensorflow-1.8.0-cp36-cp36m-macosx_10_13_x86_64.whl || exit 1

```



## 在Python脚本中加上中断



我们要在python加载`pywrap_tensorflow.so`之后用GDB/LLDB连上去, 所以为了让断点能生效，就在`import tensorflow`之后加个能让程序停

下来的代码(应该还有别的方式，我目前还没发现). 例如:



```python

import tensorflow as tf

import os



input("pid: " + str(os.getpid()) +", press enter to continue")



a = tf.constant([[1,3]])

b = tf.constant([[3],[1]])

c = tf.matmul(a, b)



with tf.Session() as sess:

    print(sess.run(c))

```



# 调试

## 使用GDB/LLDB 命令行

首先启动python程序:

```

$ python /tmp/test.py

pid: 44806, press enter to continue

```



我们看到pid是44806, 执行GDB/LLDB连上这个进程:



```

$ lldb -p 44806

(lldb) process attach --pid 44806

Process 44806 stopped

* thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGSTOP

    frame #0: 0x00007fff707ecbea libsystem_kernel.dylib`__read_nocancel + 10

libsystem_kernel.dylib`__read_nocancel:

->  0x7fff707ecbea <+10>: jae    0x7fff707ecbf4            ; <+20>

    0x7fff707ecbec <+12>: movq   %rax, %rdi

    0x7fff707ecbef <+15>: jmp    0x7fff707e3ae9            ; cerror_nocancel

    0x7fff707ecbf4 <+20>: retq

Target 0: (Python) stopped.



Executable module set to "/Users/chi/.pyenv/versions/3.6.4/Python.framework/Versions/3.6/Resources/Python.app/Contents/MacOS/Python".

Architecture set to: x86_64h-apple-macosx.

(lldb)

```



此时Python程序已经被停下来了，接下来可以设置断点:



```

(lldb) breakpoint set -n TF_NewSession

Breakpoint 1: where = _pywrap_tensorflow_internal.so`::TF_NewSession(TF_Graph *, const TF_SessionOptions *, TF_Status *) + 31 at c_api.cc:2474, address = 0x000000011191246f

(lldb) breakpoint set -f matmul_op.cc -l 458

Breakpoint 2: 6 locations.

(lldb) breakpoint list

Current breakpoints:

1: name = 'TF_NewSession', locations = 1, resolved = 1, hit count = 0

  1.1: where = _pywrap_tensorflow_internal.so`::TF_NewSession(TF_Graph *, const TF_SessionOptions *, TF_Status *) + 31 at c_api.cc:2474, address = 0x000000011191246f, resolved, hit count = 0



2: file = 'matmul_op.cc', line = 458, exact_match = 0, locations = 6, resolved = 6, hit count = 0

  2.1: where = _pywrap_tensorflow_internal.so`tensorflow::MatMulOp<Eigen::ThreadPoolDevice, float, false>::Compute(tensorflow::OpKernelContext*) + 66 at matmul_op.cc:458, address = 0x0000000115428df2, resolved, hit count = 0

  2.2: where = _pywrap_tensorflow_internal.so`tensorflow::MatMulOp<Eigen::ThreadPoolDevice, double, false>::Compute(tensorflow::OpKernelContext*) + 66 at matmul_op.cc:458, address = 0x000000011547df42, resolved, hit count = 0

  2.3: where = _pywrap_tensorflow_internal.so`tensorflow::MatMulOp<Eigen::ThreadPoolDevice, Eigen::half, false>::Compute(tensorflow::OpKernelContext*) + 66 at matmul_op.cc:458, address = 0x00000001154d14d2, resolved, hit count = 0

  2.4: where = _pywrap_tensorflow_internal.so`tensorflow::MatMulOp<Eigen::ThreadPoolDevice, int, false>::Compute(tensorflow::OpKernelContext*) + 66 at matmul_op.cc:458, address = 0x0000000115524702, resolved, hit count = 0

  2.5: where = _pywrap_tensorflow_internal.so`tensorflow::MatMulOp<Eigen::ThreadPoolDevice, std::__1::complex<float>, false>::Compute(tensorflow::OpKernelContext*) + 66 at matmul_op.cc:458, address = 0x000000011557a962, resolved, hit count = 0

  2.6: where = _pywrap_tensorflow_internal.so`tensorflow::MatMulOp<Eigen::ThreadPoolDevice, std::__1::complex<double>, false>::Compute(tensorflow::OpKernelContext*) + 66 at matmul_op.cc:458, address = 0x00000001155d3a22, resolved, hit count = 0

```



设置了两个断点，第一个是`TF_NewSession`函数，第二个是`matmul_op.cc`的第458行. `LLDB`的相关文档见[https://developer.apple.com/library/archive/documentation/General/Conceptual/lldb-guide/chapters/C3-Breakpoints.html](https://developer.apple.com/library/archive/documentation/General/Conceptual/lldb-guide/chapters/C3-Breakpoints.html).



接下来可以让程序继续运行. 在LLDB命令行下输入

```

(lldb) c

Process 44806 resuming

```



在python程序中按下回车, 可以看到LLDB已经运行到了断点, 进入了`TF_NewSession`:



```

Process 44806 stopped

* thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1

    frame #0: 0x000000011191246f _pywrap_tensorflow_internal.so`::TF_NewSession(graph=0x00007faaff47e310, opt=0x00007faafe87d530, status=0x00007faafc4dbc00) at c_api.cc:2474

   2471	TF_Session* TF_NewSession(TF_Graph* graph, const TF_SessionOptions* opt,

   2472	                          TF_Status* status) {

   2473	  Session* session;

-> 2474	  status->status = NewSession(opt->options, &session);

   2475	  if (status->status.ok()) {

   2476	    TF_Session* new_session = new TF_Session(session, graph);

   2477	    if (graph != nullptr) {

Target 0: (Python) stopped.

```



接下来就可以单步跟踪了。



## 使用IDE(VSCode)

GDB/LLDB只有命令行界面，使用比较麻烦，如果想用IDE可以使用VSCode.



### 安装插件

在MacOS上，需要运行

```

ln -s /Applications/Xcode.app/Contents/Developer/usr/bin/lldb-mi /usr/local/bin/lldb-mi

```

把`lldb-mi`放进`$PATH`里，才能和IDE进行连接。



首先安装一个叫`Native Debug`的插件:

![](native-debug.png)



然后点击debug标签下的配置按钮，创建一个debug配置(launch.json):

![](debug-config.png)



launch.json的内容如下:

```json

{

    "version": "0.2.0",

    "configurations": [

        { 

            "name": "(lldb) Attach",

            "type": "cppdbg",

            "request": "attach",

            "program": "/Users/chi/.pyenv/versions/3.6.4/Python.framework/Versions/3.6/Resources/Python.app/Contents/MacOS/Python",

            "processId": "${command:pickProcess}",

            "MIMode": "lldb"

        },

    ]

}

```



### 设置断点

在VSCode里打开tensorflow代码目录，设置断点即可，例如:

![](vscode-brpt.png)



### 进行调试

首先正常启动python程序:

```

$ python /tmp/test.py

pid: 45874, press enter to continue

```



然后在VSCode里点击运行按钮，运行刚才的"(lldb) Attach":

![](vscode-debug-select.png)

在输入框中找到目标python程序并点击(这里是python test.py的那个), 就可以进入调试模式.

进入python的窗口按下回车，让python程序继续运行，可以看到程序停在`TF_NewSession`函数上.

![](vscode-debug.png)

接下来就可以使用IDE中的按钮进行单步跟踪或者变量查看了。


