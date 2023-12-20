---
title: Python parallel processing
date: '2020-08-30'
tags:
  - Python
  - Programming
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/codescv/codescv.github.io/blob/main/notebooks/python-parallel-processing.ipynb)


I came across this function called `parallel` in [fastai](https://github.com/fastai/fastai), and it seems very interesting.

# A Simple Example

```python
from fastcore.all import parallel
```

```python
from nbdev.showdoc import doc
```

```python
doc(parallel)
```

<h4 id="parallel" class="doc_header"><code>parallel</code><a href="https://github.com/fastai/fastcore/tree/master/fastcore/utils.py#L715" class="source_link" style="float:right">[source]</a></h4><blockquote><p><code>parallel</code>(<strong><code>f</code></strong>, <strong><code>items</code></strong>, <strong>*<code>args</code></strong>, <strong><code>n_workers</code></strong>=<em><code>8</code></em>, <strong><code>total</code></strong>=<em><code>None</code></em>, <strong><code>progress</code></strong>=<em><code>None</code></em>, <strong><code>pause</code></strong>=<em><code>0</code></em>, <strong>**<code>kwargs</code></strong>)</p>
</blockquote>
<p>Applies <code>func</code> in parallel to <code>items</code>, using <code>n_workers</code></p>
<p><a href="https://fastcore.fast.ai/utils#parallel" target="_blank" rel="noreferrer noopener">Show in docs</a></p>

As the documentation states, the `parallel` function can run any python function `f` with `items` using multiple workers, and collect the results.

Let's try a simple examples:

```python
import math
import time

def f(x):
  time.sleep(1)
  return x * 2

numbers = list(range(10))
```

```python
%%time

list(map(f, numbers))
print()
```

    
    CPU times: user 0 ns, sys: 0 ns, total: 0 ns
    Wall time: 10 s

```python
%%time

list(parallel(f, numbers))
print()
```

    
    CPU times: user 32 ms, sys: 52 ms, total: 84 ms
    Wall time: 2.08 s

The function `f` we have in this example is very simple: it sleeps for one second and then returns `x*2`. When executed in serial, it takes 10 seconds which is exactly
what we expect. When using more workers(8 by default), it takes only 2 seconds.

# Dig into the Implementation

Let's see how `parallel` is implemented:

```python
parallel??
```

    [0;31mSignature:[0m
    [0mparallel[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0mf[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mitems[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0;34m*[0m[0margs[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mn_workers[0m[0;34m=[0m[0;36m8[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mtotal[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mprogress[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mpause[0m[0;34m=[0m[0;36m0[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0;34m**[0m[0mkwargs[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0;31mSource:[0m   
    [0;32mdef[0m [0mparallel[0m[0;34m([0m[0mf[0m[0;34m,[0m [0mitems[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0mn_workers[0m[0;34m=[0m[0mdefaults[0m[0;34m.[0m[0mcpus[0m[0;34m,[0m [0mtotal[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0mprogress[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0mpause[0m[0;34m=[0m[0;36m0[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m
    [0;34m[0m    [0;34m"Applies `func` in parallel to `items`, using `n_workers`"[0m[0;34m[0m
    [0;34m[0m    [0;32mif[0m [0mprogress[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m [0mprogress[0m [0;34m=[0m [0mprogress_bar[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m[0m
    [0;34m[0m    [0;32mwith[0m [0mProcessPoolExecutor[0m[0;34m([0m[0mn_workers[0m[0;34m,[0m [0mpause[0m[0;34m=[0m[0mpause[0m[0;34m)[0m [0;32mas[0m [0mex[0m[0;34m:[0m[0;34m[0m
    [0;34m[0m        [0mr[0m [0;34m=[0m [0mex[0m[0;34m.[0m[0mmap[0m[0;34m([0m[0mf[0m[0;34m,[0m[0mitems[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m
    [0;34m[0m        [0;32mif[0m [0mprogress[0m[0;34m:[0m[0;34m[0m
    [0;34m[0m            [0;32mif[0m [0mtotal[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m [0mtotal[0m [0;34m=[0m [0mlen[0m[0;34m([0m[0mitems[0m[0;34m)[0m[0;34m[0m
    [0;34m[0m            [0mr[0m [0;34m=[0m [0mprogress_bar[0m[0;34m([0m[0mr[0m[0;34m,[0m [0mtotal[0m[0;34m=[0m[0mtotal[0m[0;34m,[0m [0mleave[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m[0m
    [0;34m[0m        [0;32mreturn[0m [0mL[0m[0;34m([0m[0mr[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0;31mFile:[0m      /opt/conda/lib/python3.7/site-packages/fastcore/utils.py
    [0;31mType:[0m      function

```python
??ProcessPoolExecutor
```

    [0;31mInit signature:[0m
    [0mProcessPoolExecutor[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0mmax_workers[0m[0;34m=[0m[0;36m8[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mon_exc[0m[0;34m=[0m[0;34m<[0m[0mbuilt[0m[0;34m-[0m[0;32min[0m [0mfunction[0m [0mprint[0m[0;34m>[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mpause[0m[0;34m=[0m[0;36m0[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mmp_context[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0minitializer[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0minitargs[0m[0;34m=[0m[0;34m([0m[0;34m)[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0;31mSource:[0m        
    [0;32mclass[0m [0mProcessPoolExecutor[0m[0;34m([0m[0mconcurrent[0m[0;34m.[0m[0mfutures[0m[0;34m.[0m[0mProcessPoolExecutor[0m[0;34m)[0m[0;34m:[0m[0;34m[0m
    [0;34m[0m    [0;34m"Same as Python's ProcessPoolExecutor, except can pass `max_workers==0` for serial execution"[0m[0;34m[0m
    [0;34m[0m    [0;32mdef[0m [0m__init__[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mmax_workers[0m[0;34m=[0m[0mdefaults[0m[0;34m.[0m[0mcpus[0m[0;34m,[0m [0mon_exc[0m[0;34m=[0m[0mprint[0m[0;34m,[0m [0mpause[0m[0;34m=[0m[0;36m0[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m
    [0;34m[0m        [0;32mif[0m [0mmax_workers[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m [0mmax_workers[0m[0;34m=[0m[0mdefaults[0m[0;34m.[0m[0mcpus[0m[0;34m[0m
    [0;34m[0m        [0mself[0m[0;34m.[0m[0mnot_parallel[0m [0;34m=[0m [0mmax_workers[0m[0;34m==[0m[0;36m0[0m[0;34m[0m
    [0;34m[0m        [0mstore_attr[0m[0;34m([0m[0mself[0m[0;34m,[0m [0;34m'on_exc,pause,max_workers'[0m[0;34m)[0m[0;34m[0m
    [0;34m[0m        [0;32mif[0m [0mself[0m[0;34m.[0m[0mnot_parallel[0m[0;34m:[0m [0mmax_workers[0m[0;34m=[0m[0;36m1[0m[0;34m[0m
    [0;34m[0m        [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m__init__[0m[0;34m([0m[0mmax_workers[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m
    [0;34m[0m[0;34m[0m
    [0;34m[0m    [0;32mdef[0m [0mmap[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mf[0m[0;34m,[0m [0mitems[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m
    [0;34m[0m        [0mself[0m[0;34m.[0m[0mlock[0m [0;34m=[0m [0mManager[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mLock[0m[0;34m([0m[0;34m)[0m[0;34m[0m
    [0;34m[0m        [0mg[0m [0;34m=[0m [0mpartial[0m[0;34m([0m[0mf[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m
    [0;34m[0m        [0;32mif[0m [0mself[0m[0;34m.[0m[0mnot_parallel[0m[0;34m:[0m [0;32mreturn[0m [0mmap[0m[0;34m([0m[0mg[0m[0;34m,[0m [0mitems[0m[0;34m)[0m[0;34m[0m
    [0;34m[0m        [0;32mtry[0m[0;34m:[0m [0;32mreturn[0m [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mmap[0m[0;34m([0m[0mpartial[0m[0;34m([0m[0m_call[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mlock[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mpause[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mmax_workers[0m[0;34m,[0m [0mg[0m[0;34m)[0m[0;34m,[0m [0mitems[0m[0;34m)[0m[0;34m[0m
    [0;34m[0m        [0;32mexcept[0m [0mException[0m [0;32mas[0m [0me[0m[0;34m:[0m [0mself[0m[0;34m.[0m[0mon_exc[0m[0;34m([0m[0me[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0;31mFile:[0m           /opt/conda/lib/python3.7/site-packages/fastcore/utils.py
    [0;31mType:[0m           type
    [0;31mSubclasses:[0m     

As we can see in the source code, under the hood, this is using the [concurrent.futures.ProcessPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor) class from Python.

Note that this class is essentially different than Python Threads, which is subject to the Global Interpreter Lock.

The ProcessPoolExecutor class is an Executor subclass that uses a pool of processes to execute calls asynchronously. ProcessPoolExecutor uses the multiprocessing module, which allows it to side-step the Global Interpreter Lock but also means that only picklable objects can be executed and returned.

# Use cases

This function can be quite useful for long running tasks and you want to take advantage of multi-core CPUs to speed up your processing. For example, if you want to download a lot of images from the internet, you may want to use this to parallize your download jobs. 

If your function `f` is very fast, there can be suprising cases, here is an example:

```python
import math
import time

def f(x):
  return x * 2

numbers = list(range(10000))
```

```python
%%time

list(map(f, numbers))
print()
```

    
    CPU times: user 0 ns, sys: 0 ns, total: 0 ns
    Wall time: 1.24 ms

```python
%%time

list(parallel(f, numbers))
print()
```

    
    CPU times: user 3.96 s, sys: 940 ms, total: 4.9 s
    Wall time: 12.4 s

In the above example, `f` is very fast and the overhead of creating a lot of tasks outweigh the advantage of multi-processing. So use this with caution, and always take profiles.
