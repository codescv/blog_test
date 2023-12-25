---
title: Projects
date: 2023-12-07
---

Here are a list of open source projects I built.

# Babel Tower (2020)
[Babel Tower](https://github.com/google/gps-babel-tower) is a Python package designed to address common NLP tasks such as keyword extraction, translation, sentiment analysis, language detection, and text similarity identification. Its development was motivated by practical use cases within Google Ads, including:

1. Search Keyword Expansion: Identifying potential "missing" keywords to complement existing ad campaign strategies.
2. Topic and Sentiment Analysis: Extracting insights from app and YouTube comment data.
3. Cost-Effective Translation and Language Detection: Optimizing ad campaigns for international audiences.

While the advent of Large Language Models (LLMs) has provided alternative solutions for many of these tasks,
Babel Tower remains a valuable resource for its modularity and efficiency.


# Function Flow (2019)
[Function Flow](https://github.com/codescv/function_flow) is a workflow management layer built on top of Google Cloud Functions. 
While Google Cloud Functions offers a convenient way to run Python code in a managed service, 
there were previously no simple methods for orchestrating multiple functions into a workflow. 
In response to this need, I created Function Flow as a "Pythonic" approach to workflow management.

In short, you can wrap Python functions as tasks, specifying their dependencies:
```python
job = Job(params)

@job.task(task_id='gen_data')
def task1(*args):
    ...

@job.task(task_id='process_data', deps=['gen_data'])
def task2(*args):
    ...
```

And then fire up the workflow like this:
```python
job.start(args)
```

Here is a more complicated [example](https://github.com/codescv/function_flow/blob/main/example/src/main.py).

Function Flow is not in active development today. See Google Cloud [Workflows](https://cloud.google.com/workflows) for a GCP managed way of doing basically the same thing.


# NESGym (2016)
[NESGym](https://github.com/codescv/nesgym) was a personal project that I undertook after being captivated by the groundbreaking achievements of AlphaGo. The advent of Reinforcement Learning (RL) garnered significant attention in the machine learning community during that time. Notably, OpenAI introduced a gym environment to facilitate RL experimentation. As an avid Nintendo enthusiast, I found it intriguing to create an environment for the Nintendo Entertainment System (NES) emulator as well.