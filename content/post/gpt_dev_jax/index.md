---
title: GPT from Scratch in Jax
date: '2024-02-05'
summary: Implements GPT from scratch using Jax, reproduct the NanoGPT model from the video Let\'s build GPT: from scratch, in code, spelled out. by Andrej Karpathy
tags: 
  - DeepLearning
  - Python
  - GPT
  - Jax
links:
  - icon_pack: custom
    icon: colab
    name: Colab
    url: 'https://colab.research.google.com/github/codescv/codescv.github.io/blob/main/notebooks/gpt_dev_jax.ipynb'
---


TLDR: I'm reproducing the GPT transformer from this video [Let's build GPT: from scratch, in code, spelled out.
](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Andrej Karpathy.

The original one was trained using pytorch, but I'm using Jax.

# Some Notes about Jax

## Jax vs Pytorch differences

- Gradients are computed explicitly by ```grad(fn)(params)```
  - No `torch.no_grad()`. no gradient computed by default.
  - No `zero_gradient` and `loss.backward()`
  - You don't need to put model into `training` or `eval` modes.
- All states (params and optimizer states) are explicit and immutable.
- Model can be defined to work with a single example instead of a mini-batch.   
  - Use `vmap` to transform it to a batched version.
  - This is not mandatory; you can still define models to work with only batched data.
- Random numbers are explicit and deterministic

## GPU preallocation
Don't panic if the GPU memory usage suddenly goes up.

JAX will [preallocate](https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html) 75% of the total GPU memory when the first JAX operation is run.

# Dataset and Task

We are building a character-based language model from the tiny shakespeare dataset.

## tokenization

The tokenization for char-based LMs are straightforward: just map chars to integers and back.

```python
import os
from functools import partial
from dataclasses import dataclass

import numpy as np

import jax
import optax
from jax import jit, nn, vmap, grad, value_and_grad, device_put
from jax import numpy as jnp, random as jrandom

if not os.path.exists('input.txt'):
  !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
  print('data downloaded')
with open('input.txt', encoding='utf-8') as f:
  text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print('vocab size:', vocab_size, 'vocabulary:', ''.join(chars), )

char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

def encode(text):
  return [char_to_idx[c] for c in text]

def decode(indices):
  return ''.join([idx_to_char[i] for i in indices])
```

    vocab size: 65 vocabulary: 
     !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz

## jax dataloader, version 1

```python
# %90 train data, 10% validation data
data = jnp.array(encode(text), dtype=jnp.int32)
n = int(len(data)*0.9)
train_data = data[:n]
val_data = data[n:]

def get_batch(key, split, context_length=8, batch_size=32):
  """get a random batch of split of train or val data."""
  data = train_data if split == 'train' else val_data
  idx = jrandom.randint(key, minval=0, maxval=len(data)-context_length, shape=(batch_size,))
  xb = jnp.stack([data[i: i+context_length] for i in idx])
  yb = jnp.stack([data[i+1:i+1+context_length] for i in idx])
  return xb, yb

xb, yb = get_batch(jrandom.PRNGKey(0), 'train')
print(f'xb: {xb.shape}, yb: {yb.shape}')
print('decoded x:', decode(xb[0].tolist()), ',y:', decode(yb[0].tolist()))
del xb, yb
```

    xb: (32, 8), yb: (32, 8)
    decoded x: she's at ,y: he's at 

The function above looks right (and in fact is), but the problem is that it is extremely slow. Since I didn't find the speed problem until I started training, it took me quite some time to realize it was because of the data loading, not the actual model training.

```python
%timeit get_batch(jrandom.PRNGKey(0), 'train')
```

    436 ms ± 161 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

## jax dataloader, version 2

The orginal torch version, although looking very similar, was much faster:

```python
import torch

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(0)

def get_batch(split, context_length=8, batch_size=32):
  """get a random batch of split of train or val data."""
  data = train_data if split == 'train' else val_data
  idx = torch.randint(len(data) - context_length, (batch_size,))
  xb = torch.stack([data[i: i+context_length] for i in idx])
  yb = torch.stack([data[i+1:i+1+context_length] for i in idx])
  return xb, yb

xb, yb = get_batch('train')
print(f'xb: {xb.shape}, yb: {yb.shape}')
print('decoded x:', decode(xb[0].tolist()), ',y:', decode(yb[0].tolist()))
del xb, yb

%timeit get_batch('train')
```

    xb: torch.Size([32, 8]), yb: torch.Size([32, 8])
    decoded x: he gives ,y: e gives 
    796 µs ± 8.35 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

I tried different ways, the closes I could get with Jax was this:

```python
data = jnp.array(encode(text), dtype=jnp.int32)
n = int(len(data)*0.9)
train_data = data[:n]
val_data = data[n:]

def get_batch(key, split, context_length=8, batch_size=32):
  """get a random batch of split of train or val data."""
  data = train_data if split == 'train' else val_data
  idx = jrandom.randint(key, minval=0, maxval=len(data)-context_length, shape=(batch_size,))
  idx_batch = idx[...,None] + jnp.arange(context_length)
  xb, yb = data[idx_batch], data[idx_batch+1]
  return xb, yb

xb, yb = get_batch(jrandom.PRNGKey(0), 'train')
print(f'xb: {xb.shape}, yb: {yb.shape}')
print('decoded x:', decode(xb[0].tolist()), ', y:', decode(yb[0].tolist()))
del xb, yb

%timeit get_batch(jrandom.PRNGKey(0), 'train')
```

    xb: (32, 8), yb: (32, 8)
    decoded x: she's at , y: he's at 
    8.67 ms ± 1.32 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)

## numpy dataloader

The best solution I found was to just use numpy for data loading:

```python
import numpy as np
data = np.array(encode(text), dtype=np.int32)
n = int(len(data)*0.9)
train_data = data[:n]
val_data = data[n:]

np.random.seed(0)

def get_batch(split, context_length=8, batch_size=32):
  """get a random batch of split of train or val data."""
  data = data = train_data if split == 'train' else val_data
  idx = np.random.randint(low=0, high=len(data)-context_length, size=(batch_size,))
  xb = np.stack([data[i: i+context_length] for i in idx])
  yb = np.stack([data[i+1:i+1+context_length] for i in idx])
  return jnp.array(xb), jnp.array(yb)
```

```python
xb, yb = get_batch('train')
print(f'xb: {xb.shape}, yb: {yb.shape}')
print('decoded x:', decode(xb[0].tolist()), ',y:', decode(yb[0].tolist()))
del xb, yb

%timeit get_batch('train')
```

    xb: (32, 8), yb: (32, 8)
    decoded x: er hand. ,y: r hand.
    
    1.29 ms ± 382 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

## Why the jax version was slow

The root cause was, for some reason, Jax was much slower at generating random integers:

```python
%timeit np.random.randint(low=0, high=100000, size=(32,))
```

    16 µs ± 6.12 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

```python
%timeit torch.randint(100000, (32,))
```

    3.96 µs ± 797 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

```python
k = jrandom.PRNGKey(0)
%timeit jrandom.randint(k, minval=0, maxval=100000, shape=(32,))
```

    501 µs ± 111 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

## jax dataloader, super fast version

The problem can be solved by jitting the function:

```python
k = jrandom.PRNGKey(0)
@jit
def gen_num(key):
  return jrandom.randint(k, minval=0, maxval=100000, shape=(32,))

gen_num(k); # skip compilation

%timeit gen_num(k)
```

    76.9 µs ± 2.22 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

Here is a jax dataloader using jit:

```python
# %90 train data, 10% validation data
data = jnp.array(encode(text), dtype=jnp.int32)
n = int(len(data)*0.9)
train_data = data[:n]
val_data = data[n:]

@partial(jit, static_argnames=['split', 'context_length', 'batch_size'])
def get_batch(key, split, context_length=8, batch_size=32):
  """get a random batch of split of train or val data."""
  print(f'load {split} data')
  data = train_data if split == 'train' else val_data
  idx = jrandom.randint(key, minval=0, maxval=len(data)-context_length, shape=(batch_size,))
  idx_batch = idx[...,None] + jnp.arange(context_length)
  xb, yb = data[idx_batch], data[idx_batch+1]
  print(xb.shape, yb.shape)
  return device_put(xb), device_put(yb)
```

```python
xb, yb = get_batch(jrandom.PRNGKey(0), 'train')
print(f'xb: {xb.shape}, yb: {yb.shape}')
print('decoded x:', decode(xb[0].tolist()), ',y:', decode(yb[0].tolist()))
del xb, yb

k = jrandom.PRNGKey(0)
%timeit get_batch(k, 'train')
```

    load train data
    (32, 8) (32, 8)
    xb: (32, 8), yb: (32, 8)
    decoded x: she's at ,y: he's at 
    96 µs ± 3.83 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

# Embedding Tables

An embedding table is a table of size `(vocab_size, emb_size)`. Each row is a vector of size `emb_size` for a token in the vocabulary.

- Input: `(...)` of integers in the range `0..vocab_size-1`.
- Output: `(..., emb_size)`.

```python
@dataclass
class Embed:
  vocab_size: int
  emb_size: int

  def init(self, key):
    C = jrandom.normal(key, (self.vocab_size, self.emb_size))
    return dict(emb_table=C)

  def __call__(self, params, x, **kwargs):
    return params['emb_table'][x]
```

```python
emb = Embed(vocab_size, 3)
p_emb = emb.init(jrandom.PRNGKey(0))
x = jnp.array([0,1,2,3])
x.shape, emb(p_emb, x).shape
```

    ((4,), (4, 3))

# Dense Layers

A dense layer maps the input (linearly) into a different dimension using `weights` and `bias`.

- Input:  (..., fan_in)
- Output:  (..., fan_out)
- Params:
  - Weights: (fan_in, fan_out)
  - Bias: (fan_out)

This is a generalized form of matrix multiplication, since the input can be more than two dimensions.

For example:

```python
@dataclass
class Dense:
  fan_in: int
  fan_out: int
  bias: bool = True

  def init(self, key):
    initializer = nn.initializers.lecun_uniform()
    weight = initializer(key, (self.fan_in, self.fan_out))
    bias = jnp.zeros((self.fan_out,)) if self.bias else None
    return dict(w=weight, b=bias)

  def __call__(self, params, x, **kwargs):
    out = x @ params['w']
    if self.bias:
      out += params['b']
    return out
```

```python
dense = Dense(fan_in=3, fan_out=4)
p_dense = dense.init(jrandom.PRNGKey(0))
x = jnp.array([[1,2,3],[2,3,4]], dtype=jnp.float32)
x.shape, dense(p_dense, x).shape
```

    ((2, 3), (2, 4))

# Simple Baseline: Bigram Model

Reproduce the baseline [bigram model](https://youtu.be/kCc8FmEb1nY?t=1337):

```python
@dataclass
class BigramLM:
  vocab_size: int
  emb_size: int

  def __post_init__(self):
    self.emb = Embed(self.vocab_size, self.emb_size)
    self.net = Dense(self.emb_size, self.vocab_size)

    self.layers = {
        'emb': self.emb,
        'net': self.net
    }

  def init(self, key):
    keys = jrandom.split(key, len(self.layers))
    params = {}

    for key, name, layer in zip(keys,
                                self.layers.keys(),
                                self.layers.values()):
      params[name] = layer.init(key)
    return params

  def __call__(self, params, x, **kwargs):
    x = self.emb(params['emb'], x, **kwargs)
    x = self.net(params['net'], x, **kwargs)
    return x

  def __hash__(self):
    return hash(id(self))

  def __eq__(self, other):
    return id(self) == id(other)

@partial(jit, static_argnames=['model', 'training'])
def loss_fn(params, model, x, y, training, key=None):
  logits = model(params, x, key, training)
  loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y).mean()
  return loss

@partial(jit, static_argnames=['model', 'optimizer'])
def step_fn(params, model, opt_state, optimizer, x, y, key):
  print('first step')
  loss, grads = value_and_grad(loss_fn)(params, model, x, y, training=True, key=key)
  updates, opt_state = optimizer.update(grads, opt_state, params=params)
  params = optax.apply_updates(params, updates)
  return loss, params, opt_state

def generate(key, params, model, prefix, max_steps, context_length):
  result = []
  x = jnp.array(encode(prefix))

  for step in range(max_steps):
    step_key = jrandom.fold_in(key, step)
    # truncate x by context length
    x = x[-context_length:]
    # get last token logits
    logits = model(params, x, training=False)[-1]
    # sample next token
    next_token = jrandom.categorical(step_key, logits=logits)
    result.append(next_token.item())
    # append next token to input
    x = jnp.concatenate((x, jnp.array([next_token])))

  return decode(result)

def evaluate(key, params, model, context_length, steps=200):
  metrics = {}

  for split in ['train', 'val']:
    loss = []
    for step in range(steps):
      step_key = jrandom.fold_in(key, step)
      xb, yb = get_batch(step_key, split, context_length=context_length)
      l = loss_fn(params, model, xb, yb, training=False)
      loss.append(l)
    metrics[f'{split}_loss'] = jnp.array(loss).mean().item()

  return metrics

def train_loop(model, total_steps: int,
               context_length: int = 8,
               batch_size: int = 32,
               lr: float=1e-3):
  optimizer = optax.adamw(lr)

  # init
  root_key = jrandom.PRNGKey(0)
  model_key, train_key, sample_key, eval_key = jrandom.split(root_key, 4)
  params = device_put(model.init(model_key))
  opt_state = device_put(optimizer.init(params))

  # workaround: vmap currently doesn't support named arguments
  def model_apply(params, x, key, training: bool):
    return model(params, x, key=key, training=training)
  batch_model = vmap(model_apply, in_axes=(None, 0, None, None))

  # train
  metrics = []
  for step in range(total_steps):
    step_key = jrandom.fold_in(train_key, step)
    xb, yb = get_batch(step_key, 'train',
                       batch_size=batch_size,
                       context_length=context_length)
    step_key, _ = jrandom.split(step_key)
    loss, params, opt_state = step_fn(
        params, batch_model,
        opt_state, optimizer,
        xb, yb, step_key)

    if step % (total_steps // 10) == 0 or step == total_steps - 1:
      # evaluate
      metric = evaluate(jrandom.fold_in(eval_key, step),
                        params, batch_model, context_length)
      metrics.append(metric)
      print(f'--- evaluation {step=} ---')
      print(f'mini batch loss: {loss.item()}\n'
            f'metrics: {metric}')

      print('--- end evaluation ---')

  # final sample
  sample_result = generate(sample_key,
                           params, model,
                           prefix='\n',
                           max_steps=500,
                           context_length=context_length)
  print('Final sample:', sample_result)
  return params, metrics
```

training bigram model for 10000 steps should give a loss of roughly 2.50

```python
bigram_model = BigramLM(vocab_size, vocab_size)
bigram_params, bigram_metrics = train_loop(
    bigram_model, total_steps=10000, context_length=1)
```

    first step
    --- evaluation step=0 ---
    mini batch loss: 4.765635967254639
    metrics: {'train_loss': 4.712054252624512, 'val_loss': 4.699409484863281}
    --- end evaluation ---
    --- evaluation step=1000 ---
    mini batch loss: 2.742424488067627
    metrics: {'train_loss': 2.635215997695923, 'val_loss': 2.657052755355835}
    --- end evaluation ---
    --- evaluation step=2000 ---
    mini batch loss: 2.530101776123047
    metrics: {'train_loss': 2.5586252212524414, 'val_loss': 2.606309652328491}
    --- end evaluation ---
    --- evaluation step=3000 ---
    mini batch loss: 2.243786334991455
    metrics: {'train_loss': 2.508502244949341, 'val_loss': 2.5393943786621094}
    --- end evaluation ---
    --- evaluation step=4000 ---
    mini batch loss: 2.6591711044311523
    metrics: {'train_loss': 2.513347625732422, 'val_loss': 2.540261745452881}
    --- end evaluation ---
    --- evaluation step=5000 ---
    mini batch loss: 2.180464744567871
    metrics: {'train_loss': 2.48012638092041, 'val_loss': 2.5483450889587402}
    --- end evaluation ---
    --- evaluation step=6000 ---
    mini batch loss: 2.611126184463501
    metrics: {'train_loss': 2.4999704360961914, 'val_loss': 2.503690719604492}
    --- end evaluation ---
    --- evaluation step=7000 ---
    mini batch loss: 2.5699210166931152
    metrics: {'train_loss': 2.5017287731170654, 'val_loss': 2.520076036453247}
    --- end evaluation ---
    --- evaluation step=8000 ---
    mini batch loss: 2.6787352561950684
    metrics: {'train_loss': 2.49395751953125, 'val_loss': 2.4981493949890137}
    --- end evaluation ---
    --- evaluation step=9000 ---
    mini batch loss: 2.3774499893188477
    metrics: {'train_loss': 2.4943182468414307, 'val_loss': 2.527510404586792}
    --- end evaluation ---
    --- evaluation step=9999 ---
    mini batch loss: 2.3447186946868896
    metrics: {'train_loss': 2.4814400672912598, 'val_loss': 2.5175700187683105}
    --- end evaluation ---
    Final sample: Thel ppanthethasonncave gweroubetu, WAersofer Gir'd ll s toind;
    Wes cth whetr
    O his isatheis!
    TEORERofor d!
    PmeeswinousUTh m,
    RIUSh be s, tu thame y ienilot m l t lll.
    ORI houtheleery n t momar ppes.
    CEat inl hithyofe se, soucanonqurto fasthanone thave ndithacke cerace, t t sira pechourse hes bur aklthialighep omyoof ys,
    O fofu io t her, s cout ' hal th be, ar,
    Pouns; l of hous
    BUS:
    
    ADe se, cacit d takepevendithed.
    
    Thereasol avaventharifonomy beins w?
    
    
    Whe toral  heealol w y pule womeve ais h

# Dropout

Dropout is added at the following places:
- the end of every residual path of self attention and feedfoward
- the attention weights

```python
@dataclass
class Dropout:
  rate: float

  def init(self, key):
    return {}

  def __call__(self, x, key=None, training=False, **kwargs):
    if training:
      mask = jrandom.bernoulli(key, p=(1-self.rate), shape=x.shape)
      x = x * mask
    else:
      x = x / (1-self.rate)
    return x
```

```python
x = jrandom.normal(jrandom.PRNGKey(1), (5,5))
dropout = Dropout(0.1)
out1 = dropout(x, jrandom.PRNGKey(2), training=True)
out2 = dropout(x)

print(x)
print(out1)
print(out2)
```

    [[ 0.59333676 -0.82349354  1.1586576   0.61708856  0.5213631 ]
     [ 0.2781005  -1.2627544   0.05730288 -0.49172685 -0.35850936]
     [-1.0447503   0.1234699   1.1976635  -0.14236492 -3.7156198 ]
     [-1.6393571   0.92326057 -1.8844254  -0.96750796 -0.63999134]
     [ 0.8939773  -0.32139128 -1.1945074   2.2471828  -2.0013103 ]]
    [[ 0.59333676 -0.82349354  1.1586576   0.61708856  0.        ]
     [ 0.2781005  -1.2627544   0.05730288 -0.49172685 -0.35850936]
     [-1.0447503   0.          1.1976635  -0.14236492 -3.7156198 ]
     [-1.6393571   0.92326057 -1.8844254   0.         -0.63999134]
     [ 0.8939773  -0.32139128 -1.1945074   2.2471828   0.        ]]
    [[ 0.6592631  -0.91499287  1.2873974   0.685654    0.57929236]
     [ 0.30900055 -1.4030606   0.06366987 -0.5463632  -0.39834374]
     [-1.1608337   0.13718878  1.3307374  -0.15818325 -4.1284666 ]
     [-1.8215079   1.0258452  -2.093806   -1.0750089  -0.71110153]
     [ 0.9933081  -0.35710144 -1.3272305   2.49687    -2.2236784 ]]

# Self Attention

## Step 1: compute bow by averaging previous steps

With an input x of `(T, C)`, compute xbow (bag of words), where each step `t` of xbow is the average of x values for steps `1..t-1`.

```python
# (T, C)
T, C = 8, 2
x = jnp.arange(T * C, dtype=jnp.float32).reshape(T, C)
```

```python
x, x.shape
```

    (Array([[ 0.,  1.],
            [ 2.,  3.],
            [ 4.,  5.],
            [ 6.,  7.],
            [ 8.,  9.],
            [10., 11.],
            [12., 13.],
            [14., 15.]], dtype=float32),
     (8, 2))

Compute with for loop

```python
import numpy as np
# use numpy, because jnp arrays are immuatble
xbow = np.zeros((T, C))
for t in range(T):
  xprev = x[:t+1]
  xbow[t] = np.mean(xprev, axis=0)
xbow, xbow.shape
```

    (array([[0.        , 1.        ],
            [1.        , 2.        ],
            [2.        , 3.        ],
            [3.        , 4.        ],
            [4.        , 5.        ],
            [5.        , 6.        ],
            [6.00000048, 7.00000048],
            [7.        , 8.        ]]),
     (8, 2))

## Step 2 implement bow using a weight matrix

```python
# (T, T)
weights = jnp.tril(jnp.ones((T, T)))
weights = weights / weights.sum(axis=1, keepdims=True)
print('weights:\n', weights)
xbow = weights @ x
xbow, xbow.shape
```

    weights:
     [[1.         0.         0.         0.         0.         0.
      0.         0.        ]
     [0.5        0.5        0.         0.         0.         0.
      0.         0.        ]
     [0.33333334 0.33333334 0.33333334 0.         0.         0.
      0.         0.        ]
     [0.25       0.25       0.25       0.25       0.         0.
      0.         0.        ]
     [0.2        0.2        0.2        0.2        0.2        0.
      0.         0.        ]
     [0.16666667 0.16666667 0.16666667 0.16666667 0.16666667 0.16666667
      0.         0.        ]
     [0.14285715 0.14285715 0.14285715 0.14285715 0.14285715 0.14285715
      0.14285715 0.        ]
     [0.125      0.125      0.125      0.125      0.125      0.125
      0.125      0.125     ]]

    (Array([[0.       , 1.       ],
            [1.       , 2.       ],
            [2.       , 3.       ],
            [3.       , 4.       ],
            [4.       , 5.       ],
            [5.       , 6.0000005],
            [6.0000005, 7.0000005],
            [7.       , 8.       ]], dtype=float32),
     (8, 2))

## Step 3 implement bow use softmax + mask

This seems redundant compared to step 2, however an advantage is that in this approach the initial weights can be any real number rather than a proper probability distribution.

```python
mask = jnp.tril(jnp.ones((T, T)))
weights = jnp.zeros((T, T))
weights = jnp.where(mask, weights, float('-inf'))
print('masked weights:\n', weights)
weights = nn.softmax(weights, axis=-1)
print('normalized weights\n', weights)
xbow = weights @ x
xbow, xbow.shape
```

    masked weights:
     [[  0. -inf -inf -inf -inf -inf -inf -inf]
     [  0.   0. -inf -inf -inf -inf -inf -inf]
     [  0.   0.   0. -inf -inf -inf -inf -inf]
     [  0.   0.   0.   0. -inf -inf -inf -inf]
     [  0.   0.   0.   0.   0. -inf -inf -inf]
     [  0.   0.   0.   0.   0.   0. -inf -inf]
     [  0.   0.   0.   0.   0.   0.   0. -inf]
     [  0.   0.   0.   0.   0.   0.   0.   0.]]
    normalized weights
     [[1.         0.         0.         0.         0.         0.
      0.         0.        ]
     [0.5        0.5        0.         0.         0.         0.
      0.         0.        ]
     [0.33333334 0.33333334 0.33333334 0.         0.         0.
      0.         0.        ]
     [0.25       0.25       0.25       0.25       0.         0.
      0.         0.        ]
     [0.2        0.2        0.2        0.2        0.2        0.
      0.         0.        ]
     [0.16666667 0.16666667 0.16666667 0.16666667 0.16666667 0.16666667
      0.         0.        ]
     [0.14285715 0.14285715 0.14285715 0.14285715 0.14285715 0.14285715
      0.14285715 0.        ]
     [0.125      0.125      0.125      0.125      0.125      0.125
      0.125      0.125     ]]

    (Array([[0.       , 1.       ],
            [1.       , 2.       ],
            [2.       , 3.       ],
            [3.       , 4.       ],
            [4.       , 5.       ],
            [5.       , 6.0000005],
            [6.0000005, 7.0000005],
            [7.       , 8.       ]], dtype=float32),
     (8, 2))

## Step 4: self-attention

1. project x into 3 vectors `q, k, v`, of shape `(T, C)`
2. weights is obtained by `q @ k.T` shape: `(T, T)`
3. output is obtained by `weights @ v` shape: `(T, C)`.

```python
T, C = 8, 32
head_size = 16

key_x, key_k, key_q, key_v = jrandom.split(jrandom.PRNGKey(0), 4)
x = jrandom.normal(key_x, (T, C))

key = Dense(C, head_size, bias=False)
query = Dense(C, head_size, bias=False)
value = Dense(C, head_size, bias=False)
p_k = key.init(key_k)
p_q = query.init(key_q)
p_v = value.init(key_v)

q, k, v = query(p_q, x), key(p_k, x), value(p_v, x)

print(q.shape, k.shape, v.shape)
print('v:\n', v)

weights = q @ k.T
print('raw weights:\n', weights)

mask = jnp.tril(jnp.ones((T, T)))
weights = jnp.where(mask, weights, float('-inf'))
print('masked weights:\n', weights)
weights = nn.softmax(weights, axis=-1)
print('normalized weights\n', weights)
out = weights @ v
out, out.shape
```

    (8, 16) (8, 16) (8, 16)
    v:
     [[ 0.24330884  0.912989   -0.05216545  0.01312336 -0.03493929 -0.67756546
      -0.012977    0.50382483  0.43943962  1.9789965   0.39075887 -0.035795
      -0.3010243  -1.8513116   0.4575368  -0.937912  ]
     [ 0.47384062  0.53008753  0.7318371  -1.3311323  -0.0116291  -1.1984854
       0.8337784  -1.0575992  -1.2903666   0.42020324  0.4199374   0.44925377
       0.30578464 -0.9753333  -0.5892321  -0.17671284]
     [ 0.31038776  1.0070648  -0.55125964 -0.7630973  -0.9848412   0.12845135
       0.5860515  -1.2459748   0.6291082  -1.2544262   1.5047724  -0.600275
       0.53391063 -0.7571485  -0.6518285   0.5858216 ]
     [ 0.49194252  0.08106694 -1.1507456   1.0922581  -0.04301414  0.03573557
       0.6306269   0.38372898 -0.11793888  0.9654348  -0.1945737  -1.2260118
       0.04184315  0.30720368  0.11399549 -0.42123383]
     [ 0.9469206  -1.569232    0.5465861  -2.290162   -0.954916    0.5616087
      -0.8563687   1.8234979   0.01479638 -0.99293065  0.43147808  1.7037365
       0.840355   -0.84829473 -0.747861    1.0664277 ]
     [ 1.6956844  -0.11534239 -1.199537   -0.08711138  0.13695055 -1.3188653
      -0.47669256  1.0186015  -0.96485376 -0.29131424  0.1802854  -0.15312868
       1.0196195  -1.4811044  -0.79046    -2.3772593 ]
     [-0.09189808  0.8924726   0.7829564   0.4982675   0.40117612 -0.06166852
       0.40300483  1.882246    1.5449111  -0.07689744  1.6187903  -1.2706587
       0.6963316   0.5413357  -1.3530843  -1.2535295 ]
     [-0.0236129   0.36506453  0.705116   -1.4806137  -0.5162303  -0.4046365
       1.0652126  -1.0577483  -0.7451378  -0.5706383   1.8458076   0.515451
      -0.6588458  -0.28666526  0.47856635  0.28629684]]
    raw weights:
     [[  2.441443    -0.9126426   -0.16153085  -2.2524822   -1.8852967
       -0.47920895   2.6829202    0.05522335]
     [ -6.3328533   -4.8715577    3.9452403   -1.8688698    1.9682579
        2.1484756    2.2096958   -5.926384  ]
     [ -2.5939102    8.940504    -4.4174337    9.739806     1.2640417
       -0.8116516   -6.1588       2.777226  ]
     [ 10.034026   -10.121895    -1.3382523   -1.9290345   -0.43742776
        1.9217516    8.414541    -2.2159047 ]
     [ -2.7413313    8.008858    -4.5563555    2.287939    -2.505909
       -2.7744846   -3.926293     3.3465173 ]
     [  2.4515986   -2.6243255    2.479877    -6.965022    -0.92782754
        1.4243177    5.2777653    0.59847116]
     [  2.4854465    5.345042     0.23339605  -3.6697712   -2.8825538
       -0.78208673   1.97867      5.332842  ]
     [ -1.2250887   -2.0607347   -0.7713232   -4.3581753    3.203529
        1.9144485    5.8811836    1.2815868 ]]
    masked weights:
     [[  2.441443           -inf         -inf         -inf         -inf
              -inf         -inf         -inf]
     [ -6.3328533   -4.8715577          -inf         -inf         -inf
              -inf         -inf         -inf]
     [ -2.5939102    8.940504    -4.4174337          -inf         -inf
              -inf         -inf         -inf]
     [ 10.034026   -10.121895    -1.3382523   -1.9290345          -inf
              -inf         -inf         -inf]
     [ -2.7413313    8.008858    -4.5563555    2.287939    -2.505909
              -inf         -inf         -inf]
     [  2.4515986   -2.6243255    2.479877    -6.965022    -0.92782754
        1.4243177          -inf         -inf]
     [  2.4854465    5.345042     0.23339605  -3.6697712   -2.8825538
       -0.78208673   1.97867            -inf]
     [ -1.2250887   -2.0607347   -0.7713232   -4.3581753    3.203529
        1.9144485    5.8811836    1.2815868 ]]
    normalized weights
     [[1.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
      0.0000000e+00 0.0000000e+00 0.0000000e+00]
     [1.8826924e-01 8.1173074e-01 0.0000000e+00 0.0000000e+00 0.0000000e+00
      0.0000000e+00 0.0000000e+00 0.0000000e+00]
     [9.7872935e-06 9.9998868e-01 1.5802159e-06 0.0000000e+00 0.0000000e+00
      0.0000000e+00 0.0000000e+00 0.0000000e+00]
     [9.9998212e-01 1.7635450e-09 1.1509980e-05 6.3753073e-06 0.0000000e+00
      0.0000000e+00 0.0000000e+00 0.0000000e+00]
     [2.1370222e-05 9.9668223e-01 3.4797974e-06 3.2658281e-03 2.7042795e-05
      0.0000000e+00 0.0000000e+00 0.0000000e+00]
     [4.1202208e-01 2.5732073e-03 4.2383969e-01 3.3522334e-05 1.4036355e-02
      1.4749524e-01 0.0000000e+00 0.0000000e+00]
     [5.2064423e-02 9.0875691e-01 5.4763095e-03 1.1050044e-04 2.4279967e-04
      1.9836698e-03 3.1365402e-02 0.0000000e+00]
     [7.4525818e-04 3.2313965e-04 1.1732068e-03 3.2480635e-05 6.2464155e-02
      1.7210377e-02 9.0891141e-01 9.1399113e-03]]

    (Array([[ 0.24330884,  0.912989  , -0.05216545,  0.01312336, -0.03493929,
             -0.67756546, -0.012977  ,  0.50382483,  0.43943962,  1.9789965 ,
              0.39075887, -0.035795  , -0.3010243 , -1.8513116 ,  0.4575368 ,
             -0.937912  ],
            [ 0.43043858,  0.60217613,  0.5842335 , -1.0780503 , -0.01601769,
             -1.1004121 ,  0.6743604 , -0.76363105, -0.9646973 ,  0.71367604,
              0.41444397,  0.357934  ,  0.19154118, -1.1402531 , -0.3921577 ,
             -0.32002324],
            [ 0.47383812,  0.53009206,  0.7318274 , -1.3311182 , -0.01163087,
             -1.1984782 ,  0.83376974, -1.0575843 , -1.2903467 ,  0.42021585,
              0.41993886,  0.4492474 ,  0.30577907, -0.97534156, -0.58922195,
             -0.1767191 ],
            [ 0.2433112 ,  0.9129848 , -0.0521782 ,  0.01312131, -0.03495027,
             -0.6775516 , -0.012966  ,  0.5038039 ,  0.43943825,  1.9789529 ,
              0.39076796, -0.03580909, -0.3010125 , -1.8512852 ,  0.4575218 ,
             -0.9378912 ],
            [ 0.47390702,  0.52857417,  0.72566265, -1.3232131 , -0.011761  ,
             -1.1943913 ,  0.8330502 , -1.0527812 , -1.2864587 ,  0.42197314,
              0.41793397,  0.44380254,  0.3049249 , -0.9711592 , -0.5869176 ,
             -0.1774914 ],
            [ 0.49643576,  0.76533365, -0.42254835, -0.3664065 , -0.4250459 ,
             -0.41445535,  0.16288152, -0.14738102,  0.30227184,  0.22792372,
              0.8325052 , -0.26672527,  0.26523665, -1.3165531 , -0.21635427,
             -0.47428054],
            [ 0.44573897,  0.56216186,  0.68151104, -1.1981512 , -0.00516231,
             -1.1281157 ,  0.77179223, -0.8801495 , -1.099772  ,  0.47490412,
              0.46142116,  0.36323124,  0.28920707, -0.9730055 , -0.559394  ,
             -0.2500335 ],
            [ 0.0053035 ,  0.7165452 ,  0.73109376,  0.29351592,  0.3014376 ,
             -0.04810769,  0.31530303,  1.8311453 ,  1.3823403 , -0.14197445,
              1.5204482 , -1.0470436 ,  0.6974243 ,  0.40835452, -1.2863889 ,
             -1.1111131 ]], dtype=float32),
     (8, 16))

## Step 5: Multi Head Attention

A multi head attention is just splitting the projection matrix `Q, K, V` into n heads, do attention on each of the heads, then concatenate the results together.

A naive implementation would be:

```python
def multi_head_attn(x, dim, n_heads):
  head_size = dim / n_heads
  # each head projects x from `dim` into q, k, v of `head_size`
  heads = [self_attn(dim, head_size) for _ in range(n_heads)]
  return cat([h(x) for h in heads], dim=-1)
```

But they can also be implemented more efficiently by using matrix multiplication, avoiding the for loop:

```python
@dataclass
class SelfAttentionHead:
  dim: int
  num_heads: int
  context_length: int
  dropout: float = 0.0

  def __post_init__(self):
    self.key = Dense(self.dim, self.dim, bias=False)
    self.query = Dense(self.dim, self.dim, bias=False)
    self.value = Dense(self.dim, self.dim, bias=False)

    self.output = Dense(self.dim, self.dim, bias=True)

    self.dropout_weights = Dropout(self.dropout)

    # constant
    self.mask = jnp.tril(jnp.ones((self.context_length, self.context_length)))

  def init(self, key):
    key_k, key_q, key_v, key_o = jrandom.split(jrandom.PRNGKey(0), 4)
    params = {
        'key': self.key.init(key_k),
        'query': self.query.init(key_q),
        'value': self.value.init(key_v),
        'output': self.output.init(key_o),
    }
    return params

  def __call__(self, params, x, **kwargs):
    T, C = x.shape
    head_dim = self.dim // self.num_heads

    q = self.query(params['query'], x, **kwargs).reshape(-1, self.num_heads, head_dim)
    k = self.key(params['key'], x, **kwargs).reshape(-1, self.num_heads, head_dim)
    v = self.value(params['value'], x, **kwargs).reshape(-1, self.num_heads, head_dim)

    weights = jnp.einsum('qhd,khd->hqk', q, k)
    weights = weights * self.dim ** -0.5
    weights = jnp.where(self.mask[:T, :T], weights, float('-inf'))
    weights = nn.softmax(weights, axis=-1)
    weights = self.dropout_weights(weights, **kwargs)

    out = jnp.einsum('hqk,khd->qhd', weights, v)
    out = out.reshape(-1, self.dim)
    out = self.output(params['output'], out, **kwargs)
    return out
```

```python
T, C = 8, 16
head_size = 16

x = jrandom.normal(jrandom.PRNGKey(0), (T, C))

head = SelfAttentionHead(head_size, 2, T)
params = head.init(jrandom.PRNGKey(0))
out = head(params, x)

print('input:', x.shape, 'output:', out.shape)
```

    input: (8, 16) output: (8, 16)

# Feed Forward Network

This is just a simple MLP.

```python
@dataclass
class FeedFoward:
  input_size: int
  hidden_size: int

  def __post_init__(self):
    self.dense1 = Dense(self.input_size, self.hidden_size)
    self.dense2 = Dense(self.hidden_size, self.input_size)

  def init(self, key):
    keys = jrandom.split(key, 2)
    params = {
        'dense1': self.dense1.init(keys[0]),
        'dense2': self.dense2.init(keys[1]),
    }
    return params

  def __call__(self, params, x, **kwargs):
    x = self.dense1(params['dense1'], x)
    x = nn.relu(x)
    x = self.dense2(params['dense2'], x)
    return x
```

```python
ffn = FeedFoward(32, 64)
params = ffn.init(jrandom.PRNGKey(0))
x = jrandom.normal(jrandom.PRNGKey(0), (8,32))
out = ffn(params, x)
print(x.shape, out.shape)
```

    (8, 32) (8, 32)

# LayerNorm

Layernorm is similar to batchnorm. The only difference is that layernorm is computing mean and variance along the channel axis.

Regarding the scaling params, they work the same as BatchNorm.

```python
@dataclass
class LayerNorm:
  dim: int
  eps: float = 1e-5

  def init(self, key):
    return {
        'weight': jnp.ones(self.dim),
        'bias': jnp.zeros(self.dim)
    }

  def __call__(self, params, x, **kwargs):
    xmean = jnp.mean(x, axis=-1, keepdims=True)
    xvar = jnp.var(x, axis=-1, keepdims=True)
    x = (x - xmean) / jnp.sqrt(xvar + self.eps)
    x = x * params['weight'] + params['bias']
    return x
```

```python
x = jrandom.normal(jrandom.PRNGKey(0), (8, 32))
ln = LayerNorm(32)
p_ln = ln.init(jrandom.PRNGKey(0))
out = ln(p_ln, x)
print(x.shape, out.shape)
print(out[0].mean(), out[0].std())
```

    (8, 32) (8, 32)
    7.450581e-09 0.99999404

# Block

We can combine self-attention and feed foward to a transformer decoder block.

Then we can add many blocks to scale the model up.

```python
@dataclass
class Block:
  emb_size: int
  num_heads: int
  context_length: int
  dropout: float = 0.0

  def __post_init__(self):
    self.sa_head = SelfAttentionHead(self.emb_size,
                                     self.num_heads,
                                     self.context_length,
                                     self.dropout)
    self.ffn = FeedFoward(self.emb_size, self.emb_size*4)
    self.ln1 = LayerNorm(self.emb_size)
    self.ln2 = LayerNorm(self.emb_size)
    self.dropout1 = Dropout(self.dropout)
    self.dropout2 = Dropout(self.dropout)

  def init(self, key):
    keys = jrandom.split(key, 4)
    params = {
        'sa_head': self.sa_head.init(keys[0]),
        'ffn': self.ffn.init(keys[1]),
        'ln1': self.ln1.init(keys[2]),
        'ln2': self.ln2.init(keys[3]),
    }
    return params

  def __call__(self, params, x, **kwargs):
    x = self.ln1(params['ln1'], x)
    x = x + self.dropout1(self.sa_head(params['sa_head'], x))
    x = self.ln2(params['ln2'], x)
    x = x + self.dropout2(self.ffn(params['ffn'], x))
    return x
```

```python
block = Block(emb_size=32, num_heads=4, context_length=8)
params = block.init(jrandom.PRNGKey(0))
x = jrandom.normal(jrandom.PRNGKey(0), (8,32))
out = block(params, x)
print(x.shape, out.shape)
```

    (8, 32) (8, 32)

# TransformerStack

This is just a stack of multiple Blocks, plus a final layer norm layer.

```python
@dataclass
class TransformerStack:
  emb_size: int
  num_heads: int
  num_blocks: int
  context_length: int
  dropout: float = 0.0

  def __post_init__(self):
    self.blocks = [Block(self.emb_size,
                         self.num_heads,
                         self.context_length,
                         self.dropout) \
                   for _ in range(self.num_blocks)]
    self.ln = LayerNorm(self.emb_size)

  def init(self, key):
    keys = jrandom.split(key, len(self.blocks)+1)
    params = {}

    for key, (i, block) in zip(keys[:-1], enumerate(self.blocks)):
      params[f'block_{i}'] = block.init(key)

    params['final_ln'] = self.ln.init(keys[-1])
    return params

  def __call__(self, params, x, **kwargs):
    for i, block in enumerate(self.blocks):
      x = block(params[f'block_{i}'], x)
    x = self.ln(params['final_ln'], x)
    return x
```

```python
transformer_stack = TransformerStack(emb_size=32,
                                     num_heads=4,
                                     context_length=8,
                                     num_blocks=4)

params = transformer_stack.init(jrandom.PRNGKey(0))
x = jrandom.normal(jrandom.PRNGKey(0), (8,32))
out = transformer_stack(params, x)
print(x.shape, out.shape)
```

    (8, 32) (8, 32)

# Implement TransformerLM

Now we can put all the ideas together:
- token Embeddings `(T, emb)`
- positional Embeddings  `(T, emb)`
- transformer blocks x N  `(T, emb)`
  - self attention
  - feed foward
  - residual
  - layernorm
  - dropout
- language model head  `(T, vocab_size)`

```python
@dataclass
class TransformerLM:
  vocab_size: int
  emb_size: int
  context_length: int
  num_heads: int
  num_blocks: int
  dropout: float = 0.0

  def __post_init__(self):
    self.emb = Embed(self.vocab_size, self.emb_size)
    self.pos_emb = Embed(self.context_length, self.emb_size)
    self.transformer_stack = TransformerStack(
        emb_size=self.emb_size,
        num_heads=self.num_heads,
        num_blocks=self.num_blocks,
        context_length=self.context_length,
        dropout=self.dropout)
    self.lm_head = Dense(self.emb_size, self.vocab_size)

    self.layers = {
        'emb': self.emb,
        'pos_emb': self.pos_emb,
        'transformer_stack': self.transformer_stack,
        'lm_head': self.lm_head,
    }

  def init(self, key):
    keys = jrandom.split(key, len(self.layers))
    params = {}

    for key, name, layer in zip(keys,
                                self.layers.keys(),
                                self.layers.values()):
      params[name] = layer.init(key)
    return params

  def __call__(self, params, x, **kwargs):
    T = x.shape[0]

    token_emb = self.emb(params['emb'], x, **kwargs) # (T, emb_size)
    pos_emb = self.pos_emb(params['pos_emb'], jnp.arange(T), **kwargs) # (T, emb_size)

    x = token_emb + pos_emb # (T, emb_size)
    x = self.transformer_stack(params['transformer_stack'], x) # (T, emb_size)

    logits = self.lm_head(params['lm_head'], x, **kwargs) # (T, vocab_size)
    return logits

  def __hash__(self):
    return hash(id(self))

  def __eq__(self, other):
    return id(self) == id(other)

model = TransformerLM(
    vocab_size=vocab_size,
    # emb_size=384,
    emb_size=96,
    num_heads=6,
    num_blocks=6,
    context_length=256,
    dropout=0.2)
params, metrics = train_loop(model,
                             total_steps=10000,
                             context_length=256,
                             batch_size=64,
                             lr=3e-4)
```

    first step
    --- evaluation step=0 ---
    mini batch loss: 4.861026763916016
    metrics: {'train_loss': 4.228540420532227, 'val_loss': 4.239792823791504}
    --- end evaluation ---
    --- evaluation step=1000 ---
    mini batch loss: 2.1393415927886963
    metrics: {'train_loss': 2.1552505493164062, 'val_loss': 2.188020944595337}
    --- end evaluation ---
    --- evaluation step=2000 ---
    mini batch loss: 1.7912817001342773
    metrics: {'train_loss': 1.7964385747909546, 'val_loss': 1.9257885217666626}
    --- end evaluation ---
    --- evaluation step=3000 ---
    mini batch loss: 1.6336050033569336
    metrics: {'train_loss': 1.6405086517333984, 'val_loss': 1.8075802326202393}
    --- end evaluation ---
    --- evaluation step=4000 ---
    mini batch loss: 1.561478614807129
    metrics: {'train_loss': 1.5583171844482422, 'val_loss': 1.7364275455474854}
    --- end evaluation ---
    --- evaluation step=5000 ---
    mini batch loss: 1.5019328594207764
    metrics: {'train_loss': 1.5011539459228516, 'val_loss': 1.697832465171814}
    --- end evaluation ---
    --- evaluation step=6000 ---
    mini batch loss: 1.4700853824615479
    metrics: {'train_loss': 1.4643561840057373, 'val_loss': 1.6708776950836182}
    --- end evaluation ---
    --- evaluation step=7000 ---
    mini batch loss: 1.4350680112838745
    metrics: {'train_loss': 1.4392166137695312, 'val_loss': 1.6524511575698853}
    --- end evaluation ---
    --- evaluation step=8000 ---
    mini batch loss: 1.3840088844299316
    metrics: {'train_loss': 1.4103440046310425, 'val_loss': 1.632015347480774}
    --- end evaluation ---
    --- evaluation step=9000 ---
    mini batch loss: 1.377044677734375
    metrics: {'train_loss': 1.3942676782608032, 'val_loss': 1.627993106842041}
    --- end evaluation ---
    --- evaluation step=9999 ---
    mini batch loss: 1.4026801586151123
    metrics: {'train_loss': 1.3719760179519653, 'val_loss': 1.615073323249817}
    --- end evaluation ---
    Final sample: JULIET:
    Had then, none you are the lady so, revilly leasure,
    When, if I was you have savagined is close.
    
    VIRGILIA:
    Who's all majes they gluckere--
    
    DUKE VINCENTIO:
    What fear'd keep you to omack seeming
    Methinks his zenumzary.
    
    MENENIUS:
    Pet not:
    If me not, we are rack, the sir, perhap, his some king.
    
    MARCIUS:
    I'll commended that tell-mistroubl'd allend his
    Frarius in and house of marre one can time!
    
    Speak:
    There's needs,
    I pain the staff'd the instand and offenit he
    altience pule-wored.
    
    SAMP

## Results

steps = 10000
context_length = 8
embed_size = 32
learning_rate = 1e-3

- Bigram(Baseline) 2.47 / 2.49
- Single head: train 2.35 / val 2.37
- 2 heads: train 2.24 / val 2.27
- 4 heads: 2.19 / 2.27
- +FFN: 2.092 / 2.148
- 4 blocks:  2.059 / 2.139
- +residual connection and projection: 1.932 / 2.047
- +layernorm: 1.902 / 2.019
- scale up: 6 layers, context length 256, dim = 96, dropout = 0.2: 1.37 / 1.61
