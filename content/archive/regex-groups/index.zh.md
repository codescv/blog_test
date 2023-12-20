---
layout: post
title: 正则表达式中的各种分组方式
categories: [regex]
tags: [regex]
published: True

date: 2016-01-20
---

在使用正则表达式时，我们常常用`()`来表示分组，例如:

```python
>>> re.search(r'(a(b))', 'xaby').groups()
('ab', 'b')
```

有时我们不需要返回这个分组，只是想用括号来表示语义逻辑，例如:

```python
>>> re.search(r'(a(b|c))', 'xaby').groups()
('ab', 'b')
```

我们只想表示a的后面是b或者c，但是不想把b或者c当作一个分组返回。此时可以使用
`非捕获的分组(non-capturing group)`, 语法为`(?:)`:

```python
>>> re.search(r'(a(?:b|c))', 'xaby').groups()
('ab',)
```

如果我们希望表达的意思是`a的后面有b或c，但匹配后只返回a`，可以用`positive look ahead`, 语法为`(?=)`:
```python
>>> re.search(r'(a(?=b|c))', 'xaby').groups()
('a',)
```

类似的，如果想匹配`a的后面不是某个pattern`，可以用`negative look ahead`, 语法为`(?!)`
```python
>>> re.search(r'(a(?!c|d))', 'xaby').groups()
('a',)
```

同样，我们也可以表示`a的前面有或者没有某个pattern`, 分别为`positive look behind` 和 `negative look behind`, 语法为`(?<=)`和`(?<!)`:

```python
>>> re.search(r'((?<=a)b)', 'xaby').groups()
('b',)
>>> re.search(r'((?<!c|d)b)', 'xaby').groups()
('b',)
```
