---
title: 项目
date: 2023-12-07
---

 # Babel Tower（2020）
[Babel Tower](https://github.com/google/gps-babel-tower) 是一个 Python 软件包，旨在解决常见的 NLP 任务，如关键字提取、翻译、情感分析、语言检测和文本相似性识别。其开发的动机来自 Google Ads 中的实际用例，包括：

1. 搜索关键字扩展: 确定潜在的“缺失”关键字，以补充现有的广告活动策略。
2. 主题和情感分析: 从应用和 YouTube 评论数据中提取见解。
3. 低成本的翻译和语言检测: 快速过滤和处理多语言文本

尽管大型语言模型 (LLM) 的出现为许多此类任务提供了替代解决方案，但Babel Tower仍然在某些情况下能提供更高效的解决方案.

# Function Flow (2019)
[Function Flow](https://github.com/codescv/function_flow)是一个构建在 Google Cloud Functions 之上的工作流管理层。

虽然 Google Cloud Functions 提供了一种在托管服务中运行 Python 代码的便捷方式，但以前没有简单的方法来将多个函数编排到工作流中。为了满足这一需求，我创建了函数流作为一种“Pythonic”的工作流管理方法。

简而言之，您可以将 Python 函数包装成任务，并指定其依赖关系：
```python
job = Job(params)

@job.task(task_id='gen_data')
def task1(*args):
    ...

@job.task(task_id='process_data', deps=['gen_data'])
def task2(*args):
    ...
```

然后像这样启动工作流:
```python
job.start(args)
```

这里有一个更加复杂的[工作流例子](https://github.com/codescv/function_flow/blob/main/example/src/main.py).

Function Flow 目前处于非活跃开发中。建议参阅 Google Cloud [Workflow](https://cloud.google.com/workflows)作为替代方案.


# NESGym (2016)
[NESGym](https://github.com/codescv/nesgym) 是我在被 AlphaGo 的开创性成果所吸引后开始的一个个人项目。
当时，强化学习（RL）在机器学习界备受关注。值得注意的是，OpenAI 引入了一个强化学习实验环境来促进强化学习实验。
作为一个任粉，我觉得为任天堂红白机模拟器创建一个强化学习环境也很有趣。

![nes-gym](https://raw.githubusercontent.com/codescv/nesgym/master/images/soccer.png)