---
layout: post
title: Haskell Notes (2)
categories: [Haskell]
tags: [Haskell, FP]
published: True

---

# Monads
A monad is an  algebraic structure in category theory, and in Haskell it's used to describe computations as a sequece of steps, and to handle side effects such as state and IO. It is used to allow actions(e.g. mutable variables) to be implemented safely.

```haskell
hello :: String -> IO String
hello x =
  do
    putStrLn ("Hello, " ++ x)
    putStrLn "What's your name?"
    name <- getLine
    return name
```

A monad in Haskell consists of  a `do` keyword and a sequence of commands. To extract information from a monad, use `<-`. Monad is the way to implement IO operations. Note the return type of the hello function is `IO String`, not `String`.

In fact a `do` block is just a syntactic sugar. Won't go over too much, see more [Here](https://www.futurelearn.com/courses/functional-programming-haskell/1/steps/97094).

# Lazy
Haskell is lazy, meaning that the evaluation of expressions is delayed until actually needed.

```haskell
-- given definitions:
f x y = y
bot = bot

-- eval:
f bot 3
3

-- given definition:
ones = 1 :: ones

-- eval:
ones
[1,1,1,.....]

head(ones)
1
```

`bot` is the simplest infinite function defined. Because of the laziness, if we evaluate `f bot 3` we get `3`, but if we evaluate `f 3 bot` we get an infinite loop.

For another example, we can define the fibonacci sequence lazily:

```haskell
fibs = 1:1:(zipWith (+) fibs (tail fibs))
```

# lambda calculus
Lambda calculus is the theory behind functional programming. We can view lambda calculus as a minimum language:

```haskell
exp
    = const
    | var
    | exp exp
    | \ var -> exp
```

## variables
Variables are either `bound` or `free`. 
- In `\x -> x + y`, the variable `x` is bound and `y` is free. In
- In `a + (\ a -> 2^a) 3`, the first `a` is free and the other `a`'s are bound.

## Alpha conversion
Alpha conversion means you can change the bound variables freely without affecting the value of the expression. e.g.:

```haskell
(\x -> x + 1) 3
-- rename x to y, the expression stays the same.
(\y -> y + 1) 3
```

## Beta conversion
Beta conversion is just `applying` the lambda expression. It is done by substuting value for the parameters, i.e. the expression `(\x -> exp1) exp2` is converted to `exp1[exp2|x]`.

## Eta conversion
Eta conversion says that a function is equivalent to a lambda expression that takes an argument and applies the function to the argument, i.e.: `(\x -> f x)` is equal to `f`. As another example, `f x y = g y` is equivalent to `f x = g`.

## everything is a lambda
In fact, everything (variables, constants, lists, functions, etc.) is just syntactic sugar for a lambda expression. see [Link](https://www.futurelearn.com/courses/functional-programming-haskell/1/steps/110703) for details.




