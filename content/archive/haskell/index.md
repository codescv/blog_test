---

layout: post

title: Haskell Notes

categories: [Haskell]

tags: [Haskell, FP]

published: True



---



# defining functions

Haskell is a functional programming language. This basically means functions are first-class citizens.

Let's look at how to define a function.



{% highlight haskell %}

add x y = x + y



add 1 2

3

{% endhighlight %}



# functions are partial

In haskell, functions can be partially applied (applied with less arguments than defined)



{% highlight haskell %}

add1 = add 1



add1 2

3

{% endhighlight %}



If we apply `add` with only one argument `1`, then we get a new function `add1` which takes one argument.



# infix and prefix

In haskell, infix functions (i.e. operators in other languages) and prefix functions are treated alike. They can be transformed into each other using `()` and `` ` ` ``



{% highlight haskell %}

-- infix -> prefix

1 + 2

3

(+) 1 2

3



-- prefix -> infix

myadd x y = x + y

1 `add` 2

3

{% endhighlight %}



# let

The `let` statement provides syntactic sugar for `"defining variables"`.

{% highlight haskell %}

let x = 1

    y = 2

    add x y = x + y

in 1 + (add x y)

4

{% endhighlight %}

I put quotes around "defining variables" because x and y are not variables in other languages' sense. Instead, they are just bindings local to the statement, which is somewhat unique to haskell. The let statement can also define functions.



# closures/anonymous functions/lambdas

{% highlight haskell %}

-- the follwing are equivalent:

add x y = x + y

add = \x y -> x + y



-- the follwing are equivalent:

add 1

\x add 1 x

{% endhighlight %}



# lazy evaluation

{% highlight haskell %}

-- the cycle function

take 10 (cycle [1,2,3])

[1,2,3,1,2,3,1,2,3,1]



-- one way to define the cycle function

cycle x = x ++ cycle x

{% endhighlight %}

`cycle [1,2,3]` generates an infinite list `1,2,3,1,2,3,1,2,3,...`. Because Haskell is lazy, applying `take 10` to that list actually yields an finite list ``.



# type, type class

a type class is like a Java interface a type, a type is like a concrete Java class implementing an interface,



We use `data` to define a type.

{% highlight haskell %}

data Point = Point Int Int deriving Show



Point 1 2

{% endhighlight %}



The `Show` typeclass defines a `show` function which is used to print a data type. We can redefine this function as follows:



{% highlight haskell %}

data Point = Point Int Int

instance Show Point where

    show (Point x y) = (show x) ++ "," ++ (show y)

{% endhighlight %}



# type constructor

{% highlight haskell %}

data Maybe a = Nothing | Just a

{% endhighlight %}

We call `Maybe` a type constructor. Depending on the type of `a`, this constructor can end up creating a Maybe Int, Maybe Car, or Nothing.



# functors

The `Maybe` is an instance of `Functor` type class. A `Functor` is simply something that you can map on.



{% highlight haskell %}

class Functor f where

    fmap :: (a -> b) -> f a -> f b



-- lists are instances of functors

instance Functor [] where

    fmap = map



-- maybe's are instances of functors

instance Functor Maybe where

    fmap f (Just x) = Just (f x)

    fmap f Nothing = Nothing



fmap inc (Just 1)

{% endhighlight %}



It's easy (or not so easy) to see that `Lists` and `Maybe's` are instances of `Functors`.



# pattern matching and erlang-like case statements

As seen above, pattern matching can be used in function arguments. It can also be used in `case` statements and `let` statements.

{% highlight haskell %}

let (Point x y) = (Point 1 2) in

    x + y



head' :: [a] -> a

head' xs = case xs of [] -> error "No head for empty lists!"

                      (x:_) -> x

{% endhighlight %}



# :kind :info :type

- `:kind` shows type information on types and type classes

- `:info` shows detailed information on types and type classes.

- `:type` shows type for instances of types, and type classes for types.






