---
title: Stable Diffusion from Begginer to Master (4) Text Encoder and Tokenizer
date: '2023-12-20'
---

- date: "2022-12-16"
- toc: true 
- badges: true
- comments: true
- categories: [Deep Learning, Python]
- hide: false

In this tutorial we'll take a deeper look into the text processing components of stable diffusion - the `TextEncoder` and the `Tokenizer`.

# Setup

```python
!pip install -Uqq diffusers transformers ftfy accelerate bitsandbytes
#!pip install -Uqq triton xformers
```

# Pipeline

We use the same stable diffusion pipeline from last tutorial:

```python
import pprint
import torch
from diffusers import StableDiffusionPipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipe = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-base', revision="fp16", torch_dtype=torch.float16).to(device)
```

    Fetching 12 files:   0%|          | 0/12 [00:00<?, ?it/s]

Now we can access the text encoder and tokenizer by:

```python
pipe.tokenizer
```

    PreTrainedTokenizer(name_or_path='/root/.cache/huggingface/diffusers/models--stabilityai--stable-diffusion-2-base/snapshots/1cb61502fc8b634cdb04e7cd69e06051a728bedf/tokenizer', vocab_size=49408, model_max_len=77, is_fast=False, padding_side='right', truncation_side='right', special_tokens={'bos_token': AddedToken("<|startoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'eos_token': AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'unk_token': AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'pad_token': '!'})

```python
pipe.text_encoder
```

# Tokenizer

The `Tokenizer` does two things:

1. Breaks down a long text into "tokens" (the `tokenize` method).
2. Converts tokens into a list of integer ids with value range `0 ~ vocabsize-1` , which are indices into an embedding matrix(the `convert_tokens_to_ids` method). This is essentially just a dictionary lookup.

```python
tokens = pipe.tokenizer.tokenize("a highres photo of a woman wearing a red dress")
print(tokens)
```

    ['a</w>', 'high', 'res</w>', 'photo</w>', 'of</w>', 'a</w>', 'woman</w>', 'wearing</w>', 'a</w>', 'red</w>', 'dress</w>']

Tokens are not necessarily one-to-one with words - you can see from the above example that `highres` is broken into two words - `high` and `res</w>`. The symbol `</w>` inidicates the end of a word, so for example `of` and `of</w>` are two different tokens, the former meaning the `of` is in the middle of some other word. This clever and powerful idea allows us to process words that are not seen in the training data by breaking them into "subwords" that exist in the vocabulary.

Now we can convert the tokens to integer ids:

```python
pipe.tokenizer.convert_tokens_to_ids(tokens)
```

    [320, 1487, 934, 1125, 539, 320, 2308, 3309, 320, 736, 2595]

To allow batch processing, we always pad the ids to a fixed max length (77 in the stable diffusion case). If there are more than this number of tokens, the text gets truncated.

To know where the paddings start and end, we also construct a mask that marks the text as 1s and paddings as 0s.

To do all these processing in one go, we can use the `tokenizer()` method:

```python
text_inputs = pipe.tokenizer(
    ["a highres photo of a woman wearing a red dress",
     ' '.join(["a"] + ["very"]*100 + ["text"])],
    padding="max_length",
    max_length=pipe.tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt")
pprint.pprint(text_inputs)
```

    {'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1]]),
     'input_ids': tensor([[49406,   320,  1487,   934,  1125,   539,   320,  2308,  3309,   320,
               736,  2595, 49407,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0],
            [49406,   320,  1070,  1070,  1070,  1070,  1070,  1070,  1070,  1070,
              1070,  1070,  1070,  1070,  1070,  1070,  1070,  1070,  1070,  1070,
              1070,  1070,  1070,  1070,  1070,  1070,  1070,  1070,  1070,  1070,
              1070,  1070,  1070,  1070,  1070,  1070,  1070,  1070,  1070,  1070,
              1070,  1070,  1070,  1070,  1070,  1070,  1070,  1070,  1070,  1070,
              1070,  1070,  1070,  1070,  1070,  1070,  1070,  1070,  1070,  1070,
              1070,  1070,  1070,  1070,  1070,  1070,  1070,  1070,  1070,  1070,
              1070,  1070,  1070,  1070,  1070,  1070, 49407]])}

Comparing this result with the previous one, seems there are two addtional tokens `49406` and `49407` in the beginning and end of the integer ids list.

If we want to debug the input ids, we can also use the tokenizer to decode them:

```python
pipe.tokenizer.decode(torch.tensor([49406,   320,  1487,   934,  1125,   539,   320,  2308,  3309,   320, 736,  2595, 49407]))
```

    '<|startoftext|>a highres photo of a woman wearing a red dress <|endoftext|>'

The tokenizer conventionally adds `<|startoftext|>` and `<|endoftext|>` tokens to the sentence. These tokens help the model to learn the beginning and end of the sentence.

Let's check how many tokens there are in the vocabulary:

```python
pipe.tokenizer.vocab_size
```

    49408

What are the special tokens?

```python
pipe.tokenizer.special_tokens_map
```

    {'bos_token': '<|startoftext|>',
     'eos_token': '<|endoftext|>',
     'pad_token': '!',
     'unk_token': '<|endoftext|>'}

How does integer correspond to tokens?

```python
pipe.tokenizer.decoder[320], pipe.tokenizer.decoder[1125], pipe.tokenizer.decoder[49406], pipe.tokenizer.decoder[49407]
```

    ('a</w>', 'photo</w>', '<|startoftext|>', '<|endoftext|>')

How is an unknown word handled?

I haven't found a way to get a out-of-vocabulary token, since the vocabulary contains all possible bytes, which serves as a fall back.

```python
pipe.tokenizer.tokenize('你好, abcd, asdfhjklmn')
```

    ['ä½',
     'ł',
     'å¥',
     '½</w>',
     ',</w>',
     'ab',
     'cd</w>',
     ',</w>',
     'asdf',
     'h',
     'j',
     'kl',
     'mn</w>']

To see the mapping between integer indices and tokens, use the `get_vocab` method.

```python
pipe.tokenizer('abcd, asdfhjklmn')
```

    {'input_ids': [49406, 596, 4480, 267, 36857, 71, 73, 8498, 4057, 49407], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

```python
idx2token = {idx: token for token, idx in pipe.tokenizer.get_vocab().items() }
[idx2token[i] for i in pipe.tokenizer('abcd, asdfhjklmn')['input_ids']]
```

    ['<|startoftext|>',
     'ab',
     'cd</w>',
     ',</w>',
     'asdf',
     'h',
     'j',
     'kl',
     'mn</w>',
     '<|endoftext|>']

How are punctuations preprocessed?

Nothing in particular - as seen in the above example, punctuations like commas usually have their own token (`,</w>`).

To know more about tokenizers, check out the huggingface [tutorial](https://huggingface.co/course/chapter2/4?fw=pt).

# Text Encoder

The text encoder is a [CLIPTextModel](https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py#L765).

It is a transfomer model that takes token ids from tokenizer as input and get an embedding of text.

```python
inputs = pipe.tokenizer(["a photo of a cat", "a photo of a woman wearing a red dress"],
                        padding=True, return_tensors="pt").to("cuda")
pprint.pprint(inputs)
outputs = pipe.text_encoder(**inputs)
pprint.pprint(outputs)
```

    {'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:0'),
     'input_ids': tensor([[49406,   320,  1125,   539,   320,  2368, 49407,     0,     0,     0,
                 0],
            [49406,   320,  1125,   539,   320,  2308,  3309,   320,   736,  2595,
             49407]], device='cuda:0')}
    {'last_hidden_state': tensor([[[-0.3135, -0.4475, -0.0083,  ...,  0.2544, -0.0327, -0.2959],
             [ 0.1987, -1.6914, -0.8955,  ...,  0.4661, -0.0961, -2.1465],
             [ 1.0234, -0.7349, -2.5430,  ...,  0.8960, -0.0602, -1.0723],
             ...,
             [-0.0199, -0.2195, -0.0608,  ...,  0.1279,  0.1672, -0.1105],
             [-0.0690, -0.2585, -0.0515,  ...,  0.1525,  0.1367, -0.1448],
             [-0.0992, -0.2791, -0.0477,  ...,  0.1680,  0.1204, -0.1660]],
    
            [[-0.3135, -0.4475, -0.0083,  ...,  0.2544, -0.0327, -0.2959],
             [ 0.1987, -1.6914, -0.8955,  ...,  0.4661, -0.0961, -2.1465],
             [ 1.0234, -0.7349, -2.5430,  ...,  0.8960, -0.0602, -1.0723],
             ...,
             [ 0.2007,  0.2732, -0.4333,  ...,  1.0098, -1.6348,  0.8604],
             [ 0.2339,  2.2188, -0.5488,  ...,  0.0466, -2.1484,  0.1888],
             [-0.9775,  0.2269,  2.0020,  ...,  0.0747, -0.2448, -1.8760]]],
           device='cuda:0', dtype=torch.float16, grad_fn=<NativeLayerNormBackward0>),
     'pooler_output': tensor([[ 0.3347, -0.0175,  1.0537,  ...,  0.6938, -0.4158, -1.3076],
            [-0.9775,  0.2269,  2.0020,  ...,  0.0747, -0.2448, -1.8760]],
           device='cuda:0', dtype=torch.float16, grad_fn=<IndexBackward0>)}

So by default the input is padded to the max token length of the batch of text.

Does padding matter when calling text encoder? Let's see what happens if the inputs are padded to 77 tokens.

```python
inputs1 = pipe.tokenizer(["a photo of a cat", "a photo of a woman wearing a red dress"],
                         padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt").to("cuda")
pprint.pprint(inputs1)
outputs1 = pipe.text_encoder(**inputs)
pprint.pprint(outputs1)
```

    {'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0]], device='cuda:0'),
     'input_ids': tensor([[49406,   320,  1125,   539,   320,  2368, 49407,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0],
            [49406,   320,  1125,   539,   320,  2308,  3309,   320,   736,  2595,
             49407,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0]], device='cuda:0')}
    {'last_hidden_state': tensor([[[-0.3135, -0.4475, -0.0083,  ...,  0.2544, -0.0327, -0.2959],
             [ 0.1987, -1.6914, -0.8955,  ...,  0.4661, -0.0961, -2.1465],
             [ 1.0234, -0.7349, -2.5430,  ...,  0.8960, -0.0602, -1.0723],
             ...,
             [-0.0199, -0.2195, -0.0608,  ...,  0.1279,  0.1672, -0.1105],
             [-0.0690, -0.2585, -0.0515,  ...,  0.1525,  0.1367, -0.1448],
             [-0.0992, -0.2791, -0.0477,  ...,  0.1680,  0.1204, -0.1660]],
    
            [[-0.3135, -0.4475, -0.0083,  ...,  0.2544, -0.0327, -0.2959],
             [ 0.1987, -1.6914, -0.8955,  ...,  0.4661, -0.0961, -2.1465],
             [ 1.0234, -0.7349, -2.5430,  ...,  0.8960, -0.0602, -1.0723],
             ...,
             [ 0.2007,  0.2732, -0.4333,  ...,  1.0098, -1.6348,  0.8604],
             [ 0.2339,  2.2188, -0.5488,  ...,  0.0466, -2.1484,  0.1888],
             [-0.9775,  0.2269,  2.0020,  ...,  0.0747, -0.2448, -1.8760]]],
           device='cuda:0', dtype=torch.float16, grad_fn=<NativeLayerNormBackward0>),
     'pooler_output': tensor([[ 0.3347, -0.0175,  1.0537,  ...,  0.6938, -0.4158, -1.3076],
            [-0.9775,  0.2269,  2.0020,  ...,  0.0747, -0.2448, -1.8760]],
           device='cuda:0', dtype=torch.float16, grad_fn=<IndexBackward0>)}

We can check that padding does not affect encoder result at all:

```python
torch.equal(outputs.last_hidden_state, outputs1.last_hidden_state)
```

    True

Now we can take a closer look at the text encoding model:

```python
pipe.text_encoder.text_model
```

    CLIPTextTransformer(
      (embeddings): CLIPTextEmbeddings(
        (token_embedding): Embedding(49408, 1024)
        (position_embedding): Embedding(77, 1024)
      )
      (encoder): CLIPEncoder(
        (layers): ModuleList(
          (0): CLIPEncoderLayer(
            (self_attn): CLIPAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): GELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
          (1): CLIPEncoderLayer(
            (self_attn): CLIPAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): GELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
          (2): CLIPEncoderLayer(
            (self_attn): CLIPAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): GELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
          (3): CLIPEncoderLayer(
            (self_attn): CLIPAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): GELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
          (4): CLIPEncoderLayer(
            (self_attn): CLIPAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): GELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
          (5): CLIPEncoderLayer(
            (self_attn): CLIPAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): GELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
          (6): CLIPEncoderLayer(
            (self_attn): CLIPAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): GELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
          (7): CLIPEncoderLayer(
            (self_attn): CLIPAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): GELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
          (8): CLIPEncoderLayer(
            (self_attn): CLIPAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): GELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
          (9): CLIPEncoderLayer(
            (self_attn): CLIPAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): GELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
          (10): CLIPEncoderLayer(
            (self_attn): CLIPAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): GELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
          (11): CLIPEncoderLayer(
            (self_attn): CLIPAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): GELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
          (12): CLIPEncoderLayer(
            (self_attn): CLIPAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): GELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
          (13): CLIPEncoderLayer(
            (self_attn): CLIPAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): GELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
          (14): CLIPEncoderLayer(
            (self_attn): CLIPAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): GELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
          (15): CLIPEncoderLayer(
            (self_attn): CLIPAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): GELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
          (16): CLIPEncoderLayer(
            (self_attn): CLIPAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): GELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
          (17): CLIPEncoderLayer(
            (self_attn): CLIPAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): GELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
          (18): CLIPEncoderLayer(
            (self_attn): CLIPAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): GELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
          (19): CLIPEncoderLayer(
            (self_attn): CLIPAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): GELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
          (20): CLIPEncoderLayer(
            (self_attn): CLIPAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): GELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
          (21): CLIPEncoderLayer(
            (self_attn): CLIPAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): GELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
          (22): CLIPEncoderLayer(
            (self_attn): CLIPAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): GELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
      (final_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    )

The text model is a `CLIPTextTransformer`, described in detail in the [CLIP](https://openai.com/blog/clip/) paper.

```python
from transformers.models.clip.modeling_clip import CLIPTextTransformer
```

```python
CLIPTextTransformer??
```
