---
layout: post
title: "Scala Parallel Collections - Implementation"
date: "2017-04-13 15:18:29 +0800"
published: False
---

ForkJoinPool (implements ExecutorService)

submit(ForkJoinTask) async -> Future
execute(ForkJoinTask) async
invoke(ForkJoinTask) sync -> result

```Scala
val data = (1 to 100).toSeq
val result = data.par.reduce { (x, y) => x + y }
println(s"result: $result")
```

```Scala
def reduce[U >: T](op: (U, U) => U): U = {
  tasksupport.executeAndWaitResult(new Reduce(op, splitter) mapResult { _.get })
}

tasksupport = new ExecutionContextTaskSupport

class ExecutionContextTaskSupport(val environment: ExecutionContext =         scala.concurrent.ExecutionContext.global)
extends TaskSupport with ExecutionContextTasks

trait ExecutionContextTasks extends Tasks {
  private val driver: Tasks = executionContext match {
    // eci.executor: createDefaultExecutorService(): ForkJoinPool
    case eci: scala.concurrent.impl.ExecutionContextImpl => eci.executor match {
      case fjp: ForkJoinPool => new ForkJoinTaskSupport(fjp)
      // ...
    }
    // ...
  }
  def executeAndWaitResult[R, Tp](task: Task[R, Tp]): R = driver executeAndWaitResult task
}

class ForkJoinTaskSupport(val environment: ForkJoinPool = ForkJoinTasks.defaultForkJoinPool)
extends TaskSupport with AdaptiveWorkStealingForkJoinTasks

trait AdaptiveWorkStealingForkJoinTasks extends ForkJoinTasks with AdaptiveWorkStealingTasks

trait ForkJoinTasks extends Tasks with HavingForkJoinPool {
  def executeAndWaitResult =  {
    //...
    if (Thread.currentThread.isInstanceOf[ForkJoinWorkerThread]) {
      fjtask.fork
    } else {
      forkJoinPool.execute(fjtask)
    }
  }
}

// ForkJoinTask.java
ForkJoinTask<V> fork() {
    Thread t;
    if ((t = Thread.currentThread()) instanceof ForkJoinWorkerThread)
        ((ForkJoinWorkerThread)t).workQueue.push(this);
    else
        ForkJoinPool.common.externalPush(this);
    return this;
}

scala.concurrent.ExecutionContext.global = new ExecutionContextImpl(executor: createDefaultExecutorService())

createDefaultExecutorService(): ForkJoinPool {
  //...
  new ForkJoinPool(desiredParallelism, threadFactory, uncaughtExceptionHandler, true) {
      override def execute(runnable: Runnable): Unit = {
        val fjt: ForkJoinTask[_] = runnable match {
          case t: ForkJoinTask[_] => t
          case r                  => new ExecutionContextImpl.AdaptedForkJoinTask(r)
        }
        Thread.currentThread match {
          case fjw: ForkJoinWorkerThread if fjw.getPool eq this => fjt.fork()
          case _                                                => super.execute(fjt)
        }
      }
    }
}
```

```Scala
class Reduce(op: (U,U)=>U, pit: IterableSplitter[T <: U])
extends Accessor[Option[U], Reduce[U]] {
  var result: Option[U]

  // Task
  def leaf(prevr: Option[Option[U]]) = if (pit.remaining > 0) result = Some(pit.reduce(op))

  // Accessor
  def newSubtask(p: IterableSplitter[T]) = new Reduce(op, p)

  // Task
  override def merge(that: Reduce[U]) =
    if (this.result == None) result = that.result
    else if (that.result != None) result = Some(op(result.get, that.result.get))

  override def requiresStrictSplitters = true
}
```
