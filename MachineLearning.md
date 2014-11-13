## What Is Probability Density Function ##

Probability density function (PDF) is defined as a function on some n-dimensional real value that satisfy following constraints:

1. $p(x) \geq 0 ~~~\forall x \epsilon \mathbb{R}^2$ ==> applying PDF on any n-dimensional value should give value greater than equal to 0 
2. $\int_{\mathbb{R}^2}p(x)dx = 1$ ==>  area of under the curve should be equal to 1

For discrete value, probability density function is also referred as **probability mass function**. It gives probability of a random variable being exactly some real value. 

Note that **probability density function is different from probability distribution function**. Probability density function gives probability of a random variable having some specific value whereas probability distribution function is a cumulative probability of having some random variable $X$ having value $\leq x$. i.e.
$$probability density function  = p(X=x)$$
$$Probability~Distribution~Function=\int_{-\infty}^{x}p(x)dx$$

Summary:

* Discrete Variable
	- Probability Mass Function -- it answer what is the probability of X being exactly x
	- Cumulative Distribution Function -- it answers what is the probability of X being $\leq x$
* Continuous Variable
	- Probability Density Function -- answers what is the probability of X being in the neighborhood of x. 
	-  Probability Distribution Function -- answers what is the probability of X being less than or equal to x


## What Is a Variance-Covariance or Dispersion Matrix ##

* Mean: $\mu = \frac{\sum{x}}{n}$
* Biased Variance: $\sigma^2=\frac{1}{n}\sum_{i=1}^{n}(x-\mu)$
* Unbiased Variance: $\sigma^2=\frac{1}{n-1}\sum_{i=1}^{n}(x-\mu)$

Now to understand covariance, consider following  three distributions of two random variables X (say height) and Y (say weight).  Let say we want to denote the three relations in the below plots by a numerical value which is > 0 for plot 1,  = 0 for plot 2 and < 0 for plot 3.

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
One way to do this is move the origin of the axes in the three plots to the mean of the two random variables (this is show by intersection to two straight lines). Now all the data 
are split in four regions. If we take production x and y values in each these quadrants it will be positive in quadrant 1 and 3 and negative in quadrant 2 and 4. 

Further if we take mean of these products i.e $\frac{1}{n}\sum_{i=1}^{n}(x_i-\mu_X)*(y_i-\mu_Y)$, then it will > 0 for plot 1 as there more values in quadrant 1 and 3 then in quadrant 2 and 4. Similarly for plot 2 it will be around 0 and for plot 3 it will be less than 0. Thus we can write covariance as
$$cov(X,Y) = $\frac{1}{n}\sum_{i=1}^{n}(x_i-\mu_X)*(y_i-\mu_Y)$$

Also note that $Cov(X,Y) = \sigma^2$ i.e. covariance of X to itself is same as variance. 

Now assume we have n dimensional euclidean space i.e point is represented by a vector of size n. If We take covariance of each dimension to all the other dimension we will get a $nXn$ **symmetric matrix** where diagonal elements represents variance. This matrix is referred as variance-covariance matrix and is denoted by $\Sigma$. 

$$\Sigma = \begin{pmatrix}
 cov(x_1, x_1) & cov(x_1,x_2) & \cdots  & cov(x_1,x_n)\\ 
 cov(x_2,x_1) & cov(x_2,x_2) & \cdots & cov(x_2, x_n)\\ 
\cdots  & \cdots  & \cdots & cov(x_n,x_n) \\
\end{pmatrix}$$

Variance-covariance matrix has the following properties:

1. **Square**: It is a square matrix i.e. number of rows = number of columns
2. **Symmetric**: Values across diagonals are same. i.e val(1,3) = val(3,1) 
3. **Non negative Definite**:  Euclidean distance between two points in a n-dimensional space is given as $d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$. However in this form euclidean distance is sensitive to scale of the dimension. For instance, consider two person whose height and weight are measured in meter and kg. Based on the above equation we calculate distance between these two persons. Now let's say instead of using meter and kg we use meter and lbs as measuring unit and recalculate distance. The two computed distances will be different. This is somewhat counterintuitive as the two persons are same. To overcome this problem, euclidean distance is sometimes calculated as:
	$$d(x,y)=\sqrt{\sum_{i=1}^{n}w_i(x_i - y_i)^2}$$

	We can further rewrite above equation in matrix form as show below
	$$d(x,y)= \begin{pmatrix}
x_1-y_1 &  x_2-y_2& \cdot & x_n-y_n
\end{pmatrix} \begin{pmatrix}
w_1 & 0 & \cdot & 0 \\ 
0 & w_2 & 0 & 0 \\ 
\cdot & \cdot & \cdot & \cdot \\ 
0 & 0 & \cdot & w_n 
\end{pmatrix}\begin{pmatrix}
x_1-y_1 \\
x_2-y_2 \\
\cdot \\
x_n-y_n
\end{pmatrix}$$

	We can further generalize above linear algebra by replacing all 0 with appropriate weights:

	$$d(x,y)= \begin{pmatrix}
x_1-y_1 &  x_2-y_2& \cdot & x_n-y_n
\end{pmatrix} \begin{pmatrix}
w_{11} & w_{12} & \cdot & w_{1n} \\ 
w_{21} & w_{22} & \cdot & w_{2n}\\ 
\cdot & \cdot & \cdot & \cdot \\ 
w_{n1} & w_{n2} & \cdot & w_{nn} 
\end{pmatrix}\begin{pmatrix}
x_1-y_1 \\
x_2-y_2 \\
\cdot \\
x_n-y_n
\end{pmatrix}$$

	Now by definition distance is always $\geq$ 0. Thus any weight matrix $w$ that gives above distance $>$ 0 is called **positive definite**. Theoretically **positive definite** is defined as any square matrix A for which
	
	$$A = Positive~Definite ==> if a^TAa > 0 ~~~\forall a \neq \begin{pmatrix}
0 \\ 
0 \\ 
\cdots\\ 
0 
\end{pmatrix}$$

	Similarly we can define **positive semi definite** matrix $A_{nXn}$ as a matrix for which $a^TAa \geq 0 ~~\forall a$. Positive semi definite matrix is also known as **non negative matrix**. 

	An important characteristic of positive definite matrix is that all its eigen value are positive. This property is further useful in Gaussian distribution.

Note: Positive definite by itself indicate that the matrix is symmetric and symmetric metric by itself indicates that the matrix is square. 



## Explain Gaussian Distribution Function  ##

Gaussian distribution function is a probability density function that has the following form:
$$p(x) = N(x|\mu, \Sigma) = \frac{1}{(\sqrt{2\pi})^n|\Sigma|^\frac{1}{2}}exp\{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\} ~~~\forall ~~~x~\epsilon~\mathbb{R}^n$$

where:

* $\mu$ is mean vector
* $\Sigma$ is a variance-covariance matrix or **dispersion matrix**. It is $nXn$ matrix 
* $|\Sigma|$ is a determinant of variance-covariance matrix. Although determinant of a matrix is not necessarily always positive. If this happens then the above equation can result in non real number if  $|\Sigma| <  0$ ) or undefined value (if $|\Sigma| := 0$). This contradicts definition of PDF which has to be strictly $> 0$. Hence $|\Sigma|$ has to be strictly greater than 0. 

	As discussed in the above Variance-Covariance section, Variance-Covariance matrix is non-negative definite matrix. But for Gaussian distribution it can be show than Variance-Covariance matrix $|\Sigma|$ is a **positive definite matrix** and hence all its eigen values are positive. Now determinant of a matrix is product of its eigen values and hence it can shown that $|\Sigma| > 0$ 

## Pattern Recognition Steps ##
Pattern recognition involves two types of transformation. 

1. Measurement Space --> Feature Space: This step involves feature selection
2. Feature Space --> Decision Space: This step involves supervised or unsupervised classification. Supervised classification can be of two types:
	1. Conditional Probability density function and prior probabilities are known: Assume we want to differentiate male from female. We conduct an exhaustive survey where we randomly choose a person and record gender and height. Assume we surveyed 100 people and ended up with 60 male and 40 female. Then we have some prior information of a person being male and its P(x=male) = 0.6 and P(x=female) = 0.4. Based on the above survey we can also get distribution function for male and female. Assuming these distributions to be parametric (i.e we can represent the distribution using certain parameters)  

		Given prior probability and probability density function for male and female class now we can calculate probability of any random person to be male or female. For instance for any random person $x$, the person being male is given as $p(x|male)XP(male)$. Similarly we can calculate probability of being female. Which ever probability is higher is the identended class. 
Now if random select any person we can calculate

	1. Training sample points are known
		
Class conditional probability density function $p(x|w_i)$ where p represents probability density function, i represents ith class, $w_i$ represents probability density function's parameter for the ith class. So $p(x|w_i)$ basically means probability of point x belong to class i whose probability density function is given by $w_i$.  



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
