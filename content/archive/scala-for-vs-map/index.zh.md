---

layout: post

title: scala中的for语句和map

categories: [Scala, FP]

tags: [Scala, FP]

published: True



---



Scala中的for语句是一种语法糖。



# case 1: for (x) y

{% highlight scala %}

for (a <- x; b <- y; c <- z) expr

{% endhighlight %}



会转成：



{% highlight scala %}

x.foreach(a => y.foreach(c => z.foreach(expr)))

{% endhighlight %}



# case 2: for (x) yield y



{% highlight scala %}

for (a <- x; b <- y; c <- z) yield expr

{% endhighlight %}



会转成：



{% highlight scala %}

x.flatMap(a => y.flatMap(c => z.map(expr)))

{% endhighlight %}



# for中的类型匹配



{% highlight scala %}

for (x: String <- Seq("a", "b", null)) yield x

{% endhighlight %}



会返回`List(a, b)`。也就是说会对后面的值进行匹配，而null不会匹配`String`, 因此yield不会出`null`.



可以用`scala -Xprint:parser` 来查看编译器生成的代码，非常有用：



{% highlight scala %}

scala> for (x: String <- Seq("a", "b", null)) yield x

[[syntax trees at end of                    parser]] // <console>

package $line3 {

  object $read extends scala.AnyRef {

    def <init>() = {

      super.<init>();

      ()

    };

    object $iw extends scala.AnyRef {

      def <init>() = {

        super.<init>();

        ()

      };

      object $iw extends scala.AnyRef {

        def <init>() = {

          super.<init>();

          ()

        };

        val res0 = Seq("a", "b", null).withFilter(((check$ifrefutable$1) => check$ifrefutable$1: @scala.unchecked match {

  case (x @ (_: String)) => true

  case _ => false

})).map(((x: String) => x))

      }

    }

  }

}

{% endhighlight %}



可以明显看到后面会有一个withFilter的操作。`for (x: String <- Seq("a", "b", null)) yield x` 和 `for (x <- Seq("a", "b", null)) yield x` 的结果是不同的，这一点要特别注意.



参考问题：[http://stackoverflow.com/questions/41499441/explanation-on-scala-for-comprehension-with-option](http://stackoverflow.com/questions/41499441/explanation-on-scala-for-comprehension-with-option)










