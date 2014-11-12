## What Is a Variance-Covariance or Dispersion Matrix ##

* Mean: $\mu = \frac{\sum{x}}{n}$
* Biased Variance: $\sigma^2=\frac{1}{n}\sum_{i=1}^{n}(x-\mu)$
* Unbiased Variance: $\sigma^2=\frac{1}{n-1}\sum_{i=1}^{n}(x-\mu)$

Now to understand covariance, consider following  three distributions of two random variables X (say height) and Y (say weight). 

```R
library('MASS')
library('ggplot2')
library('grid')

covars = c(0.7, 0, -0.7)
par(mfrow = c(1, 3))
out <- NULL
counter = 1
for (covar in covars){
  d1 <- as.data.frame(mvrnorm(1000, c(1,2), matrix(c(1.0, covar, covar, 1), nrow=2)))
  p <- ggplot() + 
    geom_point(aes(x=V1, y=V2), data=d1) + 
    geom_vline(xintercept=1) + 
    geom_hline(yintercept=2) +
    xlab("height") + ylab("weight") 
  out[[counter]] = p
  counter = counter + 1
}
grid.arrange(out[[1]], out[[2]], out[[3]], ncol=3)
```
![Correlation dataset](cor1.png)

## Explain Gaussian Distribution Function ##

In order to understand Guassian or Normal distribution function we first need to understand what is probability density function. Probability density function (PDF) is defined as a function on some n-dimensional real value that satisfy following constraints:

1. $p(x) \geq 0 ~~~\forall x \epsilon \mathbb{R}^2$ ==> applying PDF on any n-dimensional value should give value greater than equal to 0 
2. $\int_{\mathbb{R}^2}p(x)dx = 1$ ==>  area of under the curve should be equal to 1.

Gaussian distribution function is a PDF that has the following form:
$$p(x) = N(x|\mu, \Sigma) = \frac{1}{(\sqrt{2\pi})^n|\Sigma|^\frac{1}{2}}exp\{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\} ~~~\forall ~~~x~\epsilon~\mathbb{R}^n$$

where:

* $\mu$ is mean vector
* $\Sigma$ is a variance-covariance matrix or **dispersion matrix**. It is $nXn$ matrix 
* $|\Sigma|$ is a determinant of variance-covariance matrix which has to be strictly $> 0$. 

Now determinant of a matrix is not necessarily always positive i.e. it is possible $|\Sigma| \leq 0$. As a result above equation can result in non real number (if $|\Sigma| <  0$ ) or undefined value (if $|\Sigma| := 0$).  However this contradicts PDF definition as $p(x) \geq 0$. Hence $|\Sigma|$ has to be strictly greater than 0. 


An important property of $\Sigma$  in the case of Gaussian probability distribution is that its **symmetric** and **positive definite** matrix.  By positive definite we mean that 




Any function that satisfy above cons

1. Gaussian Distributions
2. Determinant of a NXN Matrix
3. Variance
4. Covariance
	- Is Covariance subject to outlier
	- Can it be more than 1 or less than 1
5. Variance-Covariance Matrix
	1. Symmetric
	2. Positive Definite Matrix 

*[PDF]: Probability Density Function
