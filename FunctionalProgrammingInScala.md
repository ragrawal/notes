


*[CBV]: Call By Value
*[CBN]: Call By Name


# Week 1: Functions &amp; Evaluations

### What are the different paradigms of programming ?
1. **Imperative Programming:** Based on Von Newman’s idea, this programming style closely maps process operations to operations on memory. For instance variable dereference is same as load instruction, variable referencing is same store operation, control structure is same as jumping across memory cells. Imperative programming provides mutual variables, uses assignments.
2. **Funcational Programming:**
3. **Logic Programming:**
4. **Object-Oriented Programming:** — Orthogonal to above three such that it can be combined with any of above three paradigms. For instance, Scala offers both functional and object oriented programming options.

### What are the challenges of Imperative Programming ?
Theories have data types and operations, but never allows for mutation. For instance the two polynomial (say $2x +b$ and $3x + b$) equations are considered different. However if a programming language allows mutation then it breaks the above concept. For instance consider the following pseudo code

```scala
class Polynomial { … }
P = new Polynomial(2, 1) // creating polynomial 2x + b
P.coefficients[0] = 3 // changing coefficient from 2 to 3
```

In the above case the same memory pointer that was original referring to $latex 2x + b $ equation now refers to $latex 3x + b $. This breaks the theory.

### What is Substitution Model ?
Substitution model is based on lambda calculus and used to resolve any expression to value. It works as long as there are no side effects (such as ```c++```) where apart from returning the value an expression also changes the state of a variable.

### What are the differences between Call by Value (CBV) and Call By Name (CBN):
CBV and CBN are two different evaluation strategy. In CBV we reduce an expression to a value first and then apply the expression. In CBN we first apply the expression and then reduce it. For instance consider the following expression definition:

```scala
def test(x: Int, y: Int): Int = x*x
test(2+4, 0) // 6
```
In case of CBV, we first reduce ```2+4``` to value of 6 and then apply ```test(6, 0)```. In contrast in CBN we first apply test(2+4, 0) and then reduce ``` x = 2+4``` to value of 6. Both strategies guarantee to reduce an expression to exactly same value as long as following conditions are meet:
1. the reduced expression consists of pure function, and
2. both evaluations terminate.

Below are different situations where one will be faster than the other

1. test(3,4) # Both will take same amount of operation
2. test(3*5, 9) # CBV will be faster 3*5 = 15 will be computed only once in CBV whereas in CBN it will be computed twice.
3. test(15, 4+5) # CBN will be faster as CBV will compute 4+5 = 9 even though we are not using it.
4. test(3+4, 2*4) # both CBV and CBN will take same amount of time.

**Advantages &amp; Issues**
CBV and CBN has there own advantages and issues. CBV has the advantage that it evaluates every function arguments only once. In contrast, CBN has the advantage that a function argument is not evaluated until it is required. For instance in the above "test" definition, y will not be evaluated as its not used by the function body. This can extremely useful while developing logging framework. For instance consider when logging we often have option to set log level either to debug, info or critical. If the log level is set to critical then we don't want to evaluate debug and info expressions. In such cases if we pass parameter by name then the expression (log message) won't be evaluated until it is required.

Another advantage of CBN is that it terminates more often that CBV. For instance consider the following scenario:

```scala
def loop(x: Int): Int = loop // Non terminating function
def test(x:Int, y:Int) = x
test(3+4, loop)
```

In the above example CBV fails to terminate as it tries to reduce "loop" (non terminating function) before evaluating the function body and gets stuck. In contrast since the function body doesn't use the second parameter at all, CBN will ignore it and thereby terminates.

Even though CBN terminates more often than CBV, Scala uses CBV by default as it offers exponential performance gain as expressions are reduced to values only once. CBN can be invoked by adding =&gt; after the semicolon (ex. y : =&gt; Int).

### What’s the difference between statement and expressions ?
In a expression we don’t need to setup a variable and return a variable. Unlike in java, "If-then-else" in Scala is an expression and not a statement. If-then-else is called predicate expression.

### What is short-circuit evaluation ?
Often predicate expression do not require to resolve all the expressions in order to determine value. For instance in the following expression, `False &amp;&amp; e`, even without knowing value of “e”, we know that the predicate expression will always return False. In such cases Scala skips evaluating e and is known as “short-circuit” evaluation.

### What’s the difference between “val” and “def” ?
In case of “val”, the right hand side is evaluate at the point of definition itself. For instance consider val x = square(2). In this case x refers to 4 and not square(2). The difference between val and def becomes evident in non-terminating functions. For instance in REPL, you can type following two commands

```scala
// this terminates as functions are not evaluated at the point of definition
def loop: Boolean = loop
// this is also fine
def x = loop
// will lead to an infinite loop as right hand side will be evaluated immediately .
val x = loop
```

### What is tail Recursion ?
If the last action of function body is to call itself then function's stack frame can be reused i.e. memory footprint of the function call remains static. Such functions are called Tail Recursive functions. Below is an example of a tail recursive function. Notice that the last action of gcd function is to call itself.

```scala
def gcd(a: Int, b:Int): Int = if (b==0) a else gcd(b, a%b)
```
Often we can convert non tail recursive function to be tail recursive. For instance below are two different implementation of factorial function. Whereas the top function is non tail recursive, the second version is tail recursive and hence has much smaller footprint.

```scala
//is not tail recursive as its not only calling itself
//but taking the result and multiplying it by n
def factorial (a: Int) : Int = if (a == 0) 1 else n * factorial(n - 1)

//tail recursive implementation: We pass current product
//to the inner function
@tailrec
def factorial (a: Int): Int = {
    def iter(x: Int, acc: Int) : Int = if (x == 0) acc else iter(x-1, acc * x)
    iter(a, 1)
}
```
While tail recursion is desired it is not forced. However one can force a function to be tail recursive by adding **@tailrec** annotation (as shown above). If a function with @tailrec annotation is not tail recursive then Scala will raise an error.

# Week 2: Higher Order Functions

### What are higher order functions ?
Functions that take (1) other functions as parameters or (2) that return functions as results are called higher order functions. For instance assume we want to write a function that computes $y=\sum_{x=a}^{b}{f(x)}$. The challenge is $f$ is known only at runtime. Thus, as shown below, we have to write a higher order function that takes "f" as one of its arguments.

```scala
def sumOfInts(f: Int => Int, a: Int, b: Int) = if (a > b) 0 else f(a) + sumOfInts(f, a+1, b)

def identity(x: Int) : Int = x
def square(x: Int) : Int = x * x
deb cube(x: Int) : Int = x * x * x

sumOfInts(identity, 0, 10) // sum of 0 to 10 integer
sumOfInts(square, 0, 10) // sum of squares
sumOfInts(cube, 0, 10) // sum of cubes

//Using anonymous functions
sumOfInts(x => x, 0, 10)
sumOfInts(x => x * x, 0, 10)
sumOfInts(x => x * x * x, 0, 10)
```
Note if an anonymous function takes two arguments, wrap them in brackets i.e. `(x: Int, y: Int) =&gt; x + y`

### What is currying ?

SumOfInts above take three arguments: 1) function that needs to be applied to each integer value and 2) the lower and 3) the upper boundary within which to iterate values. However it can be also rewritten as

```scala
//Original
def sumOfInts(f: Int => Int, a: Int, b: Int) = if (a > b) 0 else f(a) + sumOfInts(f, a+1, b)

//New
def sumOfInts(f: Int => Int): (Int, Int) => Int = {
    def sumF(a: Int, b: Int): Int ={
       if (a > b) 0
       else f(a) + sumF(a+1, b)
    }
    sumF
}

val cube = sumOfInts(x => x * x * x)
cube(0, 10)
cube(0, 5)
```

The new version of sumOfInts can be interpreted as a method that takes a function and returns a function. The input function takes one integer value and returns another integer value i.e (f: Int =&gt; Int). The returned function i.e (Int, Int) =&gt; Int, takes two input values and returns a single integer value. The advantage of using the new version is we can initialize a some function and re-use with different boundary parameters.

Since higher order functions are so useful that Scala has special syntax for writing functions that returns functions as arguments. As shown below, sumOfInts can be further simplified as below. In this new simplified version we are not using any explicit internal methods and combining parameters for all the internal classes together (but separated with different brackets).

```scala
def sumOfInts(f: Int => Int)(a: Int, b: Int): Int = if (a > b) 0 else f(a) + sumOfInts(f)(a+1, b)
```
It is further possible to break down a &amp; b into individual parameters as follows: `def sumOfInts(f: Int =&gt; Int)(a: Int)(b: Int)...`. This style of definition and function application where each anonymous function takes a single parameter is called **curring**.

### Exercise: A number x is called **fixed point** of a function f if $latex f(x) = x$. Write a method to calculate fixed point of a function.
Solution: Video 2.3 —> Good example of higher order functions and currying

### What is an expression ?
An expression is a statement or a block of statements that evaluates to a single value. An expression can be one of the following

Name | Example |
---- | :------|
Identifier | x, isGoodEnough
literal | 0, 1.0, “abc”
function application | sqrt(x)
operator application |  -x, y + x
selection | math.abs
conditional expression | if (x < 0) -x else x
block | { val x= math.abs(y); x * 2 }
anonymous function | x => x + 1
[Expression Table]

### What is a definition ?
A definition can be either:
1. a functions definition i.e. def square(x: Int): Int or
2. a value definition i.e val x = square(2)

see also: [^1],[^2],[^3]

### What is a method:
Methods are functions that operating on data abstractions in the data abstraction itself. For example, a class “rational” to represent rational number can have operators such as add, subtract, etc. These functions are called as method as they operate on the data stored by an instance of class rational.

```scala
class Rational(x: Int, y: Int){
  require(y != 0, "denominator must be nonzero")
  private def gcd(a: Int, b: Int): Int = if (b == 0) a else gcd(b, a % b)   
  private val g = gcd(x, y)
  val numer = x / g
  val denom = y / g

  //additional constructor
  def this(x: Int) = this(x, 1)

  def add(that: Rational) = {
    new Rational(numer * that.denom + that.numer * denom, denom * that.denom)
  }

  def neg: Rational = new Rational(-1 * numer, denom)

  def sub(that: Rational) = add(that.neg)

  override def toString = numer + "/" + denom

}

object rationals {
  val x = new Rational(1, 3)   // x: Rational = 1/3
  val y = new Rational(5, 7)   // y: Rational = 5/7
  val z = new Rational(3, 2)   // z: Rational = 3/2
  ((x.sub(y)).sub(z))                //res0: Rational = -79/42
  x.sub(y).sub(z)                    //res1: Rational = -79/42

}
```

1. In the above Rational class we could have replaced `val numer = x` with `def numer = x`. Think how would that affected the performance. Read difference between val and def
2. We guard against invalid rational number by using "require" expression. Alternatively we could have used **assert** statement. However require and assert have differences in intent. Require is used to enforce a precondition on the caller of a function. Assert is used to check the code of the function itself.
3. We can add additional constructors that take partial values

# Week 3: Class Hierarchies
### How to implement Singleton in Scala ?
```
class SingletonExample {
     def incremenet: SingletonExample.x += 1
     def getCount: Int = SingletonExample.x
}

object SingletonExample {
     private val x = 0
}
```


### How to import packages ?
1. **Named Import**
     * `import week3.Rational`
     * `import week3.{Rational, Hello} `
2. **Wildcard Import**
     * `import week3._`

### What is traits
Similar to Java, Scala is single inheritance language i.e. at any given time a class  can only extend one superclass. Traits allows to overcome this limitation of Scala. Traits are like interfaces in Java. However they are more powerful as they can have concrete implementation. The only restriction of traits is that they canont have value parameters i.e. val x = 1. Below is an example of Trait. 

```
trait Planner {
   def height;
   def width
   def surface = height * width
}

class Square extends Shape with Planar with Movable {    // extending functionality be adding trait Planar and Movable
  ….
}
```


### How to implement union function for Binary Search Trees
Reference: [StackOverflow Discussion](http://stackoverflow.com/questions/16217304/recursive-set-union-how-does-it-work-really)

### In binary trees, which of the two alternative union method is faster and why ?
``` ((left union right) union that) incl elem ```
``` (left union (right union that)) incl elem ```

# Week 4: Types & Pattern Matching
### Describe the nature of a functions in Scala
The function type A => B is an abbreviation for the class scala.Function1[A, B] which is defined as follows. 
```scala
package scala
trait Function1[A, B] {
     def apply(x: A): B
}
```
Above "1" in "scala.Function1" symbolizes that the function essentially takes a single input as an argument. Similar there are Function2, Function3,… to Function22. Thus a function `(x: Int) => x * x` is expanded as follows

```scala
class AnonFun extends Function1[Int, Int]{
     def apply(x: Int) = x * x
     new AnonFun 
}

//Alternatively above can be written as
new Function1[Int, Int] {
     def apply(x: Int) = x * x
}
```
Based on the above understanding we can now think how internally function application works. For instance consider the following two lines of code. 
```scala
val f = (x: Int) => x *x 
f(7) //49
```
Internally the above two lines are expanded as follows
```
// val f = (x: Int) ==> x * x is expanded as 
val f = new Function1[Int, Int] { 
               def apply(x: Int) = x * x
          }

// f(7) is expanded as
f.apply(7)   // 49
```

### What are the different forms of polymorphisms in Scala ?
### Why covariance polymorphism is bad ?
Available in many languages, there are two basic types of polymorphisms:

1. **[Subtyping](http://en.wikipedia.org/wiki/Subtyping)**: Common in objected oriented programming, in this type of polymorphism a function or data type (known as subclass) extends or overwrites functionality of another datatype (known as superclass). 
2. **[Generics/Parametric](http://en.wikipedia.org/wiki/Parametric_polymorphism)** : In this type of polymorphism, a function or a data type is written generically so that it can handle values identically without depending on their types. 

The above two polymorphisms can be combined to form yet two another forms of polymorphisms: 

1. **Bounded Parametric Polymorphism**: Its an extension of Generics/Parametric polymorphism where we put restrictions on the types that can be handled. For example consider the following function definition: `def square[S <: PositiveNumber](x: S): x * x`. Thus we are saying that the definition square works only on types that are subtypes of PositiveNumber class. Similarly we can also have `S >: PositiveNumber` to say that square accepts only classes that are super type of PositiveNumber class. Alternatively once can also both lower and upper bounds as follows: `S >: LowerClass <: UpperClass`
2. **Covariant**: Variance deals with relationships between complex types. For instance if A <: B (i.e. A is subtype of B), one can question whether A[] <: B[] i.e. a whether a list of A is a subtype of list of B. In Java this is true i.e. java supports covariance. However this creates several problems in Java as each list has to store a type tag indicating the class with which a given array was initialized. For instance consider the following relationship A, B <: C (i.e class A and B are subtypes of C). Given this relationship the code below is misleading as the variable "a" although initialized with type A ended up pointing to a object of type B and therefore the last line raises error. In order to avoid this problem, Java stores a type tag that indicates the class through which a list was initialized. The other problem with covariance is that the way it is handled in Java causes **runtime exceptions** (in contrast to compile time exceptions). 

```
a = A[new A(), new A()]   // initialize a list of A
C[] c = a                           // initialize a new variable c that points to list of A 
c[0] = new B()                // since B < C, this is theoretically possible
A d = c[0]                      // this should raise error
```
Because of these reasons Scala doesn’t support covariance. Hence in the Scala above code psuedocode, at compile time, will raise **type error** at  line 2 as A[] is not subtype of C[] (even though A <: C). 

### Why Parametric Array in Java/Scala Has to Store Type Parameter for Each Element ###

See **Covariant** section above. 

### Why Pattern Matching is Scala is more powerful then using Switch & InstanceOf used in Objected Oriented Languages
Similar to "Switch" statements, pattern matching in Scala allows to match a value against several cases. Below are some examples of pattern matching in Scala that demonstrates that they are way more powerful than using a combination of switch and instanceOf operators. 

```scala
case 1 => ... //Match input expression to integer 1
case 1 | 2 | 3 ==> ... // Match input expression to either 1, 2 or 3
case i:Int ==> ... // Match input expression matches to integer type
case d:Double ==> ... //Match input expression to double type
case Nil => ... //Match an object to nil
case x :: xs => ...  // Matches a non-empty list and assigns its head to x and tail to xs 
case X() => ... // Match expression to class X initialized with no arguments
case X(left, right) => ... // Match expression to type of class X and set left and right variable
case Address(_,_,city, state, zip) ==> .... //Match expression to type of Class Address and assign city, state and zip variable

```
References:

1. [Playing with Scala's Pattern Matching](http://kerflyn.wordpress.com/2011/02/14/playing-with-scalas-pattern-matching/)
2. [The Point of Pattern Matching in Scala](http://www.artima.com/scalazine/articles/pattern_matching.html)
3. [Scala Pattern Matching & Decomposition](http://ascendant76.blogspot.com/2012/10/scala-pattern-matching-decomposition.html)

### List Collection ###
1. List in Scala are represented as Linked List. Hence prepending an item to list is faster than appending at the end of the list as in the case of appending an element one has to traverse the whole list 
2. Similar to Arrays, Lists are homogenous i.e. all elements should be of same type. 
3. As compared to Arrays in Java, Lists differ in two ways. First they are immutable. Second they are recursive (whereas Arrays are flat). 
4. `x :: xs` gives a new list with the first element x, followed by elements of xs.
5. Fundamental operations on Lists are: head, tail, isEmpty

### List Pattern Matching ###
**Types**
1. Nil Pattern:  `case Nil => `
2. Cons Pattern:  `p :: px => ...`
3. List Pattern: `List(p1, ...,pn) => ...` same as `p1 :: ... :: pn :: Nil`
** Examples** 

Pattern | Explanation |
--------- | :------|
`1::2::xs` | Match List that starts with 1 and 2
`x :: Nil` | Match list of length 1
`List(x)` | Same as x :: Nil
`List()` | Match empty list. Same as ` Nil `
`List(2 :: xs)` | Matches a list that has only one element which is another list that starts with 2 

		
### Operator Ending With Colon (":") ###
Operators ending with colon (":") are treated differently in scala. As compare to other operators that are usually associated to left, operators ending with colon are associated to the right.  i.e.  `A :: B :: C is interpreted as A :: (B :: C)`

# Week 5 #





# References:

[^1]: [Scala Syntax Primer](http://jim-mcbeath.blogspot.com/2008/09/scala-syntax-primer.html)
[^2]: [Difference between statements & expressions](http://boldradius.com/blog-post/VBbp8jIAADIAlv6D/expressions-vs-statements-in-scala)
[^3]: [Difference between Functions & Methods](http://jim-mcbeath.blogspot.com/2009/05/scala-functions-vs-methods.html)

[
[^6]: [Scala API](http://www.scala-lang.org/api/current)
[^7]: [Scala By Example](http://www.scala-lang.org/docu/files/ScalaByExample.pdf)

