
## Exponential Constant (e)

* Discovered by Jacob Benroulli, e is also known as "Mathematical Constant", "Napier's Constant", "Euler's Number" [althought don't confuse with Euler's Constant]
* e is a special number who natural log is equal to 1 i.e $ln(e) = 1$
* e can also derived as sum of infinite series $e = \sum_{n=0}^{\infty}\frac{1}{n!}=\frac{1}{1} + \frac{1}{1 \cdot 2} + \frac{1}{1 \cdot 2 \cdot 3}$
* e is an irrational number that can not be represented as fraction of two rational numbers. 
* e is the limit of $(1 + 1/n)^n$ as n approaches infinity. This is used for compound interest computation. 


## Log

$$log(x) ==y~iff~x == 10^y $$


* Returns value (y) that raised to power 10 returns original value (x). For instance: since $10^2 = 100$ therefore $log(100) = 2$
* Not defined for $x <= 0$



## ln

$$log(x) ==y~iff~x == e^y $$

* Returns value (y) that raised to power $e=2.71828...$ returns original value (x). For instance since $e^{1.386...} = 4 => ln(4) = 1.386...$
* Returns higher value than log.
* Not defined for $x <= 0$

## exp

- Returns output value (y) by raising input value (x) to the power of $e=2.71828$. 
- Range: (0, $+\infty$)
- Domain: all real values 

## exp Properties

* $e^x \dot e^y = e^{(x+y)}$
* $\frac{e^x}{e^y} = e^{(x-y)}$
* $(e^x)^y = e^{xy}$
* $ln(e^x)=x$
* $e^{ln(x)} = x$
* $e^0 = 1$
* $e^{-a} = \frac{1}{e^a}$
* $e^{i\pi}=-1$
	
	
## Geometric Mean (GM)

$$ GM = \left( \prod_{i=1}^n{x_i}\right)^{1/n}$$

* __CAGR__ is computed using Geometric Mean
* Returns value lower than arithmetic mean but higher than Harmonic Mean
* Used when multiplying numbers make sense. For instance when describing proportional growth (such as computing avg. growth rate). 
* Doesn't apply to negative numbers as square root of negative number will lead to an imaginary number
* Geometric Interpretation: If we convert a rectangle to a square by keep its permiter same then the length of the sides of square will represent airthmetic mean. If we keep the area same then the length of the sides of square will represent geometric mean. 



## Harmonic Mean

$$ H = \frac{n}{\sum_{i=1}^{n}\frac{1}{x_i}} = \left( \frac{\sum_{i=1}^{n}{x_i^{-1}}}{n} \right)^{-1}$$

* HM is reciprocal of the airthmetic mean of the reciprocals of the given set of observations. 
* Harmonic mean tends strongly towards the least element in the dataset and returns value less than geometric mean.
* Used for computing 
	* F-Score
	* average speed \(when distance travelled is same\)
* Weighted HM = $\frac{\sum{w_i}}{\sum{\frac{wi}{x_i}}} = \left(\frac{\sum{w_ix_i^{-1}}}{\sum{w_i}}\right)^{-1}$

## Variance

$$ Population~Variance: \sigma^2 = \frac{\sum\left(x_i - \mu\right)^2}{N} = \frac{\sum{x_i}^2}{N} - \mu^2$$

$$ Sample~Variance: \sigma^2 = \frac{\sum\left(x_i - \bar{x}\right)^2}{N} = \frac{\sum{x_i}^2}{N} - \frac{N\bar{x}^2}{N-1}$$

* In the case of sample variance, we divide the numerator by n-1 instead of n so as to get unbiased estimate of the population parameter. This is known as **Bessel Correction**

## Bessel Correction

When calculating sample variance we use n-1 instead of n. This is known as Bessel Correction. Proof of Bessel correction is as follows:

Expected discrepancy between unbiased and biased variance is given as:

$$E[\sigma^2-s_{biased}^2] = E\left[\frac{1}{n}\sum(x_i-\mu)^2 - \frac{1}{n}\sum(x_i-\bar{x})^2\right]
$$

$$ = E\left[(\bar{x}-\mu)^2\right] $$
$$ = Var(\bar{x})$$

Based on Central Limit Theorem, we know that $Var(\bar{x}) = \sigma^2/n$ . Thus
$$E[\sigma^2-s_{biased}^2] = \sigma^2/n $$

So we can write biased estimator as 
$$ s_{biased}^2 = \sigma^2 - \sigma^2/n = \frac{(n-1)}{n}\sigma^2 $$

and therefore unbiased estimate is given as

$$ s^2_{unbiased} = \frac{n}{n-1}s^2_{biased} $$ 
$$ s^2_{unbiased} = \frac{n}{n-1}\times\frac{\sum(x_i-\bar{x})^2}{n} $$ 
$$ s^2_{unbiased} = \frac{\sum(x_i-\bar{x})^2}{n-1}$$ 


## Multinomial Logistic Classification

$$l = \frac{1}{N}\sum{D(wX+b, L)}$$
where 

* $l = Loss$
* $L = one~hot~encoded~target~variable$
* D(...) = Cross Entropy



1. Multinomial -- Generates scores for different categories
2. Logistic -- Uses soft max to convert scores to probabilities
3. classification -- compares probabilities to one hot encoding using cross entropy


## Cross Entropy ?
	
$$D(S, L) = \sum_i{L_i log(S_i)}$$
	

- Its a way to compare two probability vectors
- Cross entry is not symmetric. 


## Soft Max / Normalized Exponential ? 

$$ p_j = \frac{e^{z_j}}{\sum_{k=1}^{K}e^{z_k}}$$

* Used to convert real values to probability scores
* Its different from regular normalization because of the following reasons mentioned overe [here](http://cs231n.github.io/linear-classify/#softmax)

### Reasons to use exp
1. Range of exp is non-negative numbers. Thus by applying exp we are making sure that all values are $>=0$. 
2. We cannot use log function instead of exp because log is not defined for negative values. 

### Don't Forget This

- if you multiply scores by 10 then you will get extreme probability scores i.e. values near 0 and 1
- If you divide scores by 10 then you will get uniform probability distribution. 
- **Implementation Details**: The intermediate terms $e^{z_j}$ and $\sum_{k=1}^{K}e^{z_k}$ may be very large due to the exponentials. Dividing large numbers can be numerically unstable, so it is important to use a normalization trick. 

	$$ \frac{e^{z_j}}{\sum_{k=1}^{K}e^{z_k}} = \frac{Ce^{z_j}}{C\sum_{k=1}^{K}e^{z_k}} = \frac{e^{z_j + log(c)}}{\sum_{k=1}^{K}e^{z_k + log(C)}}$$
	where
	$$ log C = - max(z_1, z_2, ..., z_K) $$

	This simply states that we should shift the values inside the vector ff so that the highest value is zero
 
 **Note**: It can be shown that softmax is a generalization of sigmoid function and hence from software building perspective, its always safer to use softmax than sigmoid as it can handle more than two classes. 

## Interest Rate Computation


|Term | Formula | Description & Usage |
|-----|---------|---------------------|
|Simple|$P\dot(1+r\times n)$| Fixed, non-growing return |
|Compound (Annual) | $P\dot(1+r)^n$ | Changes each year |
|Compound (n times per year) | $P\dot(1+r/n)^{nt}$ | Changes each month/week/day (savings account) |
|Continuows Growth | $P\times e^{rt}$ | changes each instance (radioative decay, temperature)|


## Rollup Compounding Interest Rate

Convert 2% daily interest rate to annualized compounding interest rate

$$ (1+\frac{2}{100})^{30}-1$$

* Step 1: Convert to percentage
* Step 2: Add 1 to get growth rate
* Step 3: Raise to rollup period 
* Step 4: Separate one to interest rate for rolled up period
* Step 5: Multiply by 100 to get interest as percentage. 

## Central Limit Theorem
allows to estimate population parameters based on sample statistics. It basically stats that

1. Mean of sampling distribution of sample mean is equal to population mean i.e. $\mu_x = \mu$
2. Standard deviation of sampling distribution of sample mean $(\sigma_x)$ is equal to standar deviation of population dividec by the square root of sample size i.e. $\sigma_x = \sigma/\sqrt{n}$

## Variance Sum Law
Variance of the sum or difference of two independent variables is equal to the sum of their variances:

$$\sigma_{x_1 \pm x_2}^2 = \sigma_{x_1}^2 + \sigma_{x_2}^2 $$


## Normal Distribution

$$N(x; \mu, \sigma^2) = \frac{1}{\sqrt{2 \pi \sigma^2 }} e^{\frac{1}{2}(\frac{x-\mu}{\sigma})^2}$$

* $Mean = \mu$
* $Variance = \sigma^2$
* $\frac{x-\mu}{\sigma} = Z$  -- indicates how far a value from mean in terms of standard deviation. 

## 68-95-99.7

Empirical Law of Normal Distributation. It states that 

* One standard deviation from mean covers 68.2% of the normal distribution plot
* Two standard deviation from mean covers 95.5% of the normal distribution plot
* Three standard deviation from mean covers 99.7% of the normal distribution plot. 

## Binomail Distribution

It's a distribution of the number of successes in n indepenent trials, with probability p of success in each trial. **Its a generalization of bernoulli distribution.**

* Notation: $X \approx B(n,p)$
* PDF: $P(x=k) = \binom{n}{k} p^k(1-p)^{n-k}$
* Mean: $\mu = np$
* Variance: $\sigma^2 = np(1-p)$

## Bernoulli Distribution

Bernoulli distribution is a discrete probability distribution of a random variable that takes value 1 with success probability p and value 0 with failure probability of $q = 1- p$. **Its a special case of binomial distribution where n = 1**

* Notation: $X \approx Ben(p)$
* PDF: $P(x=k) = p^k(1-p)^{1-k}~~~\epsilon~k = {0, 1}$
* Mean: $\mu = p$
* Variance: $\sigma^2 = p(1-p)$


## LOcal regrESSion (LOESS)

* Steps
    1. Choose a smoothing parameter: The smoothing parameter, s, is a value in (0, 1] that represents the proportion of observations to use for local regression. 
    2. Find the k nearest neighbors to $x_0$. 
    3. Assign weights to the nearest neighbors using the following formula:
        * $w_i = \frac{32}{5}(1-(\frac{d_i}{D})^3)^3$
        * $D$ - largest distance in the neighborhood
        * $d_i$ is the distance between ith point and $x_0$
    4. Perform local weighted regression: The points in the local neighborhood of $x_0$ are used to fit and score a LOESS model at $x_0$. 

## Bias Variance Tradeoff

|  | Bias | Variance |
|-----|---------|---------------------|
| Also Known as | Underfitting | Overfitting |
| Training Error | High | Low |
| Test Error | High | High | 

**Strategies to fix Bias Problem:**

* Try larger set of features
* Try decreasing regularization
* Note: _Getting more data won't help_

**Strategies to fix Variance Problem**

* Try reducing number of features 
* Try increasing regularization parameter
* Note: _Getting more data might help_

## Effect of Data Size On Bias/Variance Tradeoff

## Logistic Function Or Logistic Cruve

Its a type of Sigmoid function 

$$f(x) = \frac{L}{1+e^{-k(x-x_0)}}$$

* $x_0$ = x-value of the sigmoid's midpoint
* L = curve's maximum value 
* k = steepness of the curve

## Standard Logistic Function 

$$f(x)=\frac{1}{1+e^{-x}}$$

* Derived from logistic function where $k=1$, $L=1$ and $x_0=0$, i.e. 

**Derivative**:

$$f(x) = \frac{1}{1+e^{-x}} = \frac{e^x}{1+e^x}$$
$$\frac{d}{dx}f(x) = \frac{e^x \cdot (1+e^x) - e^x \cdot e^x}{(1+e^x)^2}$$
$$\frac{d}{dx}f(x) = \frac{e^x}{(1+e^x)^2} = f(x)(1-f(x))$$

Also it can be show that $1-f(x) = f(-x)$. Thus $\frac{d}{dx}f(x) = \frac{d}{dx}f(-x)$

## Sigmoid Function

Its a bounded differential real function that is defined for all input values and has a non-negative derivative at each point. 

1. Bounded: output is bounded
2. differential: continuously differentiable
3. real: output is real value (note 1st point its also bounded)
4. defined for all input values: domain is $-\infty$ to $infty$
5. non-negative derivative: curve is monotonically increasing.
    * non-negiatve by itself means monotonically increasing. 

* Logistic function is a form of Sigmoid function.    
* Often standard logistic function is referred as sigmoid function. But note that sigmoid function or sigmoid is a family of functions that satisfy above constraints and logistic function is just one type of sigmoid function. Other examples o
* Other examples of Sigmoid Function are: Hyperbolic tangent, arctangent, Gudermannian, etc. 
     

## Logit function

$logit(p) = log(\frac{p}{1-p})$

* Logit is inverse of logistic function
* Its defined for value between 0 and 1 as log of negative value and for values < 0 or > 1, $P/(1-P)$ will become negative and hence undefined. 
* When P represents probability, $P/(1-P)$ represents odds. So logit function for probability returns log odds value. 


## Logit Vs Sigmoid

* Sigmoid is used to convert values in $(-\infty, +\infty)$ to (0,1). Whereas logit is used to convert values from (0, 1) to $(-\infty, +\infty)$

## Tanh

$$Tanh(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}}$$

* Also generates S shaped curve
* aka: hyperbolic tangent function
* It is a rescaling of the logistic sigmoid, such that its output range from -1 to 1 (there's horizontal stretching as well)

$$tanh(x) = 2logit^{-1}(2x) - 1$$

## Why Tanh gives better performance compared to sigmoid 

Tanh rescales the sigmoid function such that its output ranges from -1 to 1 (instead of 0 to 1). Hence its better able to center the data. 

Second, over the same range of values tanh generates much higher derivative values as compared to sigmoid and as a result the model is able to reach the optimal point faster. 

![Derivative of Sigmoid Function](https://i.stack.imgur.com/3knqy.png)
![Derivative of Tanh Function](https://i.stack.imgur.com/CRxi3.png)

## Rectifier Linear Unit (RELU)

![Rectifier Linear Unit](http://cs231n.github.io/assets/nn1/relu.jpeg)

$$relu(x) = max(0, x)$$

* For any value less than 0, its value is 0 otherwise x. 
* Nonlinear curve.
* Found to be very useful function for image recognition

## why ReLU works better than sigmoid for deep learning


## Logistic Regression

* **Hypothesis:** $h_\theta(x) = P(y=1|x;\theta) = \frac{1}{1+e^{-\theta^TX}} = \frac{1}{1+e^{-a}} = sigmoid(a)$
* where $a = \theta^TX = \sum_{i=0}^{n} x_i \times \theta_i$
* **Cost Function:** $J(\theta) = \frac{1}{m}\sum{\left[-y^ilog(h_\theta(x^i)+(1-y)(-log(1-h_\theta(x^i)))\right]}$
* **Update Function:** $\theta_j := \theta_j - \frac{\alpha}{m}\sum_{i=1}\left(h_\theta(x^i)-y^i\right)x_j^i$
* Prediction = 1 if $p(y|x) > 0.5$ else 0

Logistict regression is a type of neuron

* The cost function is convex and hence gaurantee global optimum. 
* For sample size less than 500, logistic regression tends to overestimate coefficients. Also it is recommended to have atleast 10 data points per independent variable. 

## Regularized Logistic Regression

* **Cost Function:** $J(\theta) = \frac{1}{m}\sum{\left[-y^ilog(h_\theta(x^i)+(1-y)(-log(1-h_\theta(x^i)))\right]} + \frac{\lambda}{2m}\sum_{i=1}^{n}\theta_j^2$
* **Update Function:** $\theta_j := \theta_j - \frac{\alpha}{m}\sum_{i=1}\left(h_\theta(x^i)-y^i\right)x_j^i + \frac{\lambda}{m}\theta_j$

## Difference between linear and logistic regression

* Linear regression is used for predicting continuous values where as logicstic regression is used for classification
* Linear regression output is unbounded and can range from $-\infty$ to $\infty$. Output of logistic regression is bounded between 0 and 1. 
* Linear regression assumes that input variables are linearly related to the target variable and that the 

## Intution behind Logistic Regression Cost Function

* If $y=1$, we want to assign higher penalty as predicted value moves from 1 to 0. This intutition is captured by $-log(h_\theta(x))$ curve. 
* Similarly, if $y=0$, we want to assigner higher penality as predicted values moves from 0 to 1. This intutition is captured by $-log(1-h_\theta(x))$. 

We can combined the two curves into a single function as follows
$y(-log(h_\theta(x)) + (1-y)(-log(1-h_\theta(x))$. 

At any given time only one of the two term is active. When $y=0$, the first term is zero and when $y=1$, the second term is zero. 


## Threshold Neuron (McCulloch and Pitts Neuron)

$$a = \left\{\begin{matrix}
0 & if & w\theta < threshold \\
1 & if & w\theta > threshold
\end{matrix}\right.$$

* Neuron fires 0 or 1 depending on whether the weighted sum of inputs is greater than threshold or not. Other way to write the above equation is by moving threshold on the other side i.e. $w\theta - b > 0$.

Challenges of Threshold Neuron

1. arbitrarily jumps from 0 to 1 with small changes to weights. 
2. There is uncertainity on what should be output if the weighted sum of inputs is equal to threshold. 
3. Being a step function, the equation is not differential and thereby makes neural network difficult to train. 

## Perceptron (Sigmoid Neuron)

Rosenblatt (1958) extended the idea of Threshold Neuron by changing step function to continous differential sigmod function.

## Backpropagation Derivation


## Amdahl's Law

$$speedup = \frac{1}{\frac{parallel}{\#nodes} + serial}$$

Provides a closed form solution to determine speedup of a database cluster. Above __parallel__ is the fraction of the query (read) processing time and serial is the fraction of the update processing time. This model does not consider logn running queries and deviations in the query processing time, which can be introduced by interdependencies of queries. 

## Random Variable

A random variable is essentially a function that maps all points in the sample space to real numbers.

## Kalman Filter

* Named after Rudolf E Kalman, its an optimal estimation algorithm that predicts
* assumes all variables are gaussian distributed random variables.

## Gaussian Distribution

* Any linear function of a normally distributed random process (variable) is also normally distributed random process. In particular if $X \sim N(\mu, \sigma^2)$ and $Y=aX + b$, then $Y \sim N(a\mu + b, a^2\sigma^2)$.  
* If two normally distributed random processes ($X_1$ and $X_2$) are independent then $X_1 + X_2 \sim N(\mu_1 + \mu_2, \sigma^2_1 + \sigma_2^2)$ 

## Autocorrelation

Its correlation of a random variable with itself over time. Formally the autocorrelation of a random signal $X(t)$ is defined as:

$$R_x(t_1, t_2) = E[X(t_1)X(t_2)]$$

## Wiener-Khinchine Relation

## Relation between limit and derivative

Derivative of a function say $y = ax^2$ represents tangent at a certain point along the curve represented by $ax^2$. 

Tangent at a point along the cruve represents rate of change of y w.r.t. to x. 

Using method of increments, we can show that as rate of change along x approaches zero, the rate of change along y is given as $2ax$ i.e 

$$ \lim{\Delta{x} \rightarrow 0}\frac{\Delta{y}}{\Delta{x}} = 2ax $$

which is same as 

$y' = 2ax $ or $\frac{\partial{y}}{\partial{x}} = 2ax$

## PCA

* The central idea of PCA is to retrain maximum variance in the dataset with minimal number of dimensions. This is achieved by transforming given dataset to a new vector space (principal components) which are uncoorelated. 

## Spectral Clustering

* One of the technique to partition a space based on affinity of points to other points. In the case of Spectral clustering the absolute location of the point doesn't matter but its affinity to other points is important. 


## Cobra Effect

* British started the bounty for catching cobra in order to end cobra infestation. However over the years cobra population increased because people started breeding cobra for the bounty. 


## Modularity of Network

* Modularity is the fraction of the edges that fall within the given groups minus the expected fraction of edges that would be in a group in a random graph where the degree of nodes is same. 
* $$ Q = \frac{1}{2m}\sum_{vw}\left[ A_{vw} - \frac{k_vk_w}{2m}\right]\delta(c_v,c_w) $$

where 

* $A_{vw}$ be an element of the adjacency matrix of the network thus:

$$ A_{vw} = \left\{\begin{matrix} 1 & \text{if vertex v and w are connected}\\ 0 & \text{otherise}
\end{matrix}\right. $$

* $m = \frac{1}{2}\sum_{vw}A_{vw}$ is the number of edges in the the graph. 

* $k_v = \sum_{w}A_{vw}$ is the degree of vertext v

The first term $\frac{1}{2m}\sum_{vw}A_{vw}\delta(c_v,c_w)$ defines the fraction of edges that fall within communities. The second term $k_vkw/2m$ describes the probability of an edge existing between vertices v and w if connections are made random but respecting vertex degree. 

## Alternative Forms of Modularity

$$ Q = \frac{1}{2m}\sum_{vw}\left[ A_{vw} - \frac{k_vk_w}{2m}\right]\delta(c_v,c_w) $$

Let's define 
$$ e_{ij} = \frac{1}{2m}\sum_{vw}A_{vw}\delta(c_v,i)\delta(c_w,j) $$
which is the fraction of edges that join vertices in community i to vertices in community j, and

$$ a_i = \frac{1}{2m}\sum_vk_v\delta(c_v, i) $$
which is the fraction of ends of edges that are attached to vertices in community i. Then writing $\delta(c_v, c_w) = \sum_i \delta(c_v, i)\delta(c_w, i)$ we have,

$$ Q = \frac{1}{2m}\sum_{vw}\left[ A_{vw} - \frac{k_vk_w}{2m}\right]\sum_i \delta(c_v, i)\delta(c_w, i)$$

$$ = \sum_i\left[\frac{1}{2m}\sum_{vw}A_{vw}\delta(c_v, i)\delta(c_w, i) - \frac{1}{2m}\sum_vk_v\delta(c_v, i)\frac{1}{2m}\sum_wk_w\delta(c_w, i)\right] $$

$$ = \sum_i(e_{ii} - a_i^2) $$

Reference: Finding Community Structure In Very Large Networks


## What's the difference between $R^2$ and $\bar{R^2}$ and Predicted R Square
* $R^2$ explains how well the model fits the real data and its value ranges in between 0 and 1 (or -1 and 1). 
* Adjusted $R^2$ ($\bar{R^2}$) is an unbiased estimate of R square that only increases when a new variable improves the performance of model more than is expected by changed. Adjusted R square can be negative and is always less than R square. 
* Predicted R square explains show well the model predicts responses for new observations. Predicted R square can help avoid over fitting
* 

## Linear Regression
**Cost Function:** is Mean Square Error
$$J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(y^i - h_\theta(x^i))^2$$

**Partial Derivatives**:
$$\frac{\partial J(\theta)}{\partial \theta_j} = -\frac{1}{m}\sum_{i=1}^{m}(y^i - h_\theta(x^i))x^i_j$$


## Pseudornadom Number Generator (PRNG)

* An algorithm to generate sequence of numbers whose properties approximate the properties sequences of random numbers. 
* Also known as deterministic random bit generator (DRBG)

## Middle Square Method 

* Developed by Jon Von Neumann in 1949, it is a method to generate random number. 
* Algorithm: 
	1. Take n digit number	
	2. Square it
	3. Take middle digits as randon number and as seed for next iteration. For instance  if n = 1024, $n^2 = 1048576$ to 01048576, $n' = 0485$

```
seed_number = int(input("Please enter a four digit number:\n[####] "))
number = seed_number
already_seen = set()
counter = 0

while number not in already_seen:
    counter += 1
    already_seen.add(number)
    number = int(str(number * number).zfill(8)[2:6]) # zfill adds padding of zeroes
    print(f"#{counter}: {number}")

print(f"We began with {seed_number}, and"
      f" have repeated ourselves after {counter} steps"
      f" with {number}.")
```

* For a generator of n-digit numbers, the period can be no longer than $8^n$
* Problems:
	* **Zero Mechanism**: If the middle n digits are all zeroes, the generator then outputs zeroes forever. 
	* If the first half of a number in the sequence is zeroes, the subsequent numbers will be decreasing to zero. This happens too frequently for this method to be of practical use. 
	* **Repeating Cycles**: Also can get stuck on a number other than zero. For n = 4, this occurs with the values 0100, 2500, 3792, and 7600. Other seed values form very short repeating cycles, e.g., 0540 → 2916 → 5030 → 3009 → 0540. These phenomena are even more obvious when n = 2, as none of the 100 possible seeds generates more than 14 iterations without reverting to 0, 10, 50, 60, or a 24 ↔ 57 loop.


## Weyl Sequence (PRNG)

* Fixes defects of Middle Square Method by preventing convergence to zero and repeating cycle problem by adding a weyl sequence to square. The middle is extracted by shifting right 32 bit. 


## Ridge Regression 

* It adds penalty ($\sum_i\theta_i^2$) to Sum of Square Errors to reduce overfitting
* It has a closed form solution as adding the penalty term makes hte matrix invertible even in situations when you have lot more features then examples. 

## Difference Between Lasso and Ridge Regression
Assume $\theta_1 + \theta_2 = 10$. What are the different positive values $\theta_1$ and $\theta_2$ can have ?

In case of ridge regression, we penalize square of coefficient values. Hence, the best coefficients will be $\theta_1 = 5; \theta_2 = 5$ as the sum of squares will be 50. 

Ridge regression has closed form solution as 


## Metrics Related to Confusion Matrix 

**Model Perspective**

* Precision = True Positives / (Predicted Positives)      
* False Omission Rate (FOR) = False Negative / Predicted Negative
* False Discovery Rate (FDR) = False Positives / Predicted Positives
* Negative Predictive Value (NPV) = True Negative / Predicted Negative


**Data Perspective**

* TPR = Sensitivity = Recall = Power = Probability of Detection = True Positives / (Actual Positives)
* False Negative Rate (FNR) = Miss Rate = False Negative / Actual Positives
* Specificity (SPC) = Selectivity = True Negative Rate (TNR) = True Negative / Actual Negative

**Overall**

* $F_1$ Score = 2PR / (P + R)
* Accuracy = (True Positives + True Negatives)/(Total Population)



## Momentum Gradient Descent

**Regular Gradient Descent**: $\theta^{k+1} = \theta^{k} - \alpha \Delta{f^k({\theta^k})}$

**Momentum Gradient Descent**

$$z^{k+1} = \beta z^{k} + \Delta{f^k({\theta^k})}$$
$$\theta^{k+1} = \theta^{k} - \alpha z^{k+1}$$


## Metric to evlauate Ordinal Classification


## Momentum Based Gradient Descent

Takes exponential average of gradients

1. Compute exponential average gradient: $g = \beta g + (1-\beta)\frac{\partial{X}}{\partial{\theta}}$
2. Update parameeters: $\theta = \theta - \eta g$


## RMSProp

* Variant of momentum that automatically computes learning rate for each components of the parameter space 

1. Compute gradient for the given parameters ($\theta$): $g = \frac{\partial{X}}{\partial{\theta}}$

2. Compute exponential average of gradent square:$v = \beta v + (1-\beta) g^2$


3. Compute learning rate for each component of the parameter space: $w = \frac{\eta}{\sqrt{v} + \epsilon}g$

4. update parameters: $\theta = \theta - w \theta$

Default values: 

* $\beta$ = 0.9
* $\eta$ = 0.1
* $\epsilon$ = 1e-10


## Metropolis-Hastings Sampling
* Form of Markov Chanin Monte Carlo Simulation
* Works well in high dimensional space
* Requires a simple distribution called the proposal distribution to help draw samples from an intractable posterior distribution. 
* To decide if new parameters $(\theta')$ should be accepted or rejected, the following ratio should be > random number between 0 and 1
	$$\frac{P(\theta'|D)}{P(\theta|D)} = \frac{P(D|\theta')P(\theta')}{P(D|\theta')P(\theta')} = \frac{\prod_i{P(d_i|\theta')P(\theta')}}{\prod_i{P(d_i|\theta)P(\theta)}} \approx \frac{\prod_i{P(d_i|\theta')}}{\prod_i{P(d_i|\theta)}}$$
	


## Gibbs Sampling
* Works well in low dimesional space.  

## Rejection Sampling
* Works well in low dimesional space.  


## Disjoint Probability

Two events are disjoint if $P(A \cap B) = 0$. 



## Permutaiton and Combination

$$nPk = \frac{n!}{(n-k)!}$$
$$nCk = \frac{n!}{(n-k)! k!} = \frac{nPk}{k!} = \frac{nPk}{kPk}$$ 

* Permutation : Order Matters
* Combination:
	* Order doesn't matter
	* It's Permutation divided by number of ways you can organize k things.

**Question:** How many ways P1 can select 4 cards from a deck of 52 cards and pass the remaining deck to P2 and then P2 selects 3 cards. 

1. P1 can select 4 cards from 52 cards in 52C4 ways. 
2. P2 can select 3 cards from 48 cards in 48C3 ways. 
3. Thus, the total number of ways is 52C4 X 48C3 
4. Note that this is different from 52C7 and 52C7 is wrong answer. Remember that combinaion is permutatiton divided by the number of cards selected cards be organized. Now P1 can organize his her cards in 4! and P2 can organize his/her carrds in 3! ways. Note that $4! * 3! \neq 7!$.  


## Random Probability Tips

* Combinatation is Permutation corrcted for number of ways to organize selected stuff
* With and Without replacement simply means independent or dependent probabilities, respectively. When select 4 cards from a deck of 52 cards with replacement, the probability for picking each card is independent. 


## Union of Probabilities 

* If events are disjoints: $P(\bigcup_{i=1}^nA_i)  = \sum_{i=1}^n{P(A_i)}$
* if events are dependent: $$P(\bigcup_{i=1}^nA_i)  = \sum_{i}P(A_i) - \sum_{i<j}P(A_iA_j) + \sum_{i<j<k}P(A_iA_jA_k) - \cdots + (-1)^{n+1}P(A_1 ... A_n)$$

example:

* $P(A \cup B) = P(A) + P(B) - P(A \cap B)$
* $P(A \cup B \cup C) = P(A) + P(B) + P(C) - P(A \cap B) - P(A \cap C) - P(B \cap C) + P(A \cap B \cap C)$

This method of computing probabilities is also known as "Inclusion-Exclusion" formula.


## Boole's Inequality

The inclusion-exclusion formula gives the probability of a union of events in terms of probabilities of intersections of the various subcollections of these events. Because this expression is rather complicated and probabilities of intersection may be unknown or hard to compute, it is useful to know that there are simple bounds known as boole's inequality
$$P(\bigcup_{i=1}^n A_i) \leq \sum_{i=1}^NP(A_i)$$

Using boole's inequality and that fact that $P(A \cap B) \geq P(A) + P(B) - 1$, it can be also shown that 

$$ P(\bigcap_{i=1}^n A_i) \geq \sum_{i=1}^nP(A_i) - (n - 1)$$


## Conditional Probability

$$P(A|B) = \frac{\#(AB)}{\#B} = \frac{P(AB)}{P(B)}$$

Probability of A with outcome space reduced to B. Interpretations of conditional probability:
* chance of A if B is known to have occurred
* long run relative frequency of A's among trails that produce B

Also known $P(A^c|B) = 1 - P(A|B)$ i.e. conditional probability satisfy rules of probability. 

## Rules of Conditional Probability

* Division: $P(A|B) = \frac{P(AB)}{P(B)} = \frac{\#AB}{\#B}$
* Product: $P(AB) = P(A|B)P(B) = P(B|A)P(A)$
* Average Rule: $P(A) = P(A|B_1)P(B_1) + \cdots + P(A|B_n)P(B_n)$
* Bayes Rule: $P(B_i|A) = \frac{P(A|B_i)P(B_i)}{P(A)}$


## Averaging Conditional Probability

$$ P(A) = P(A|B)P(B) + P(A|B^c)P(B^c) $$ 
In general: $$P(A) = \sum_{i=1}^n{P(AB_i)} = \sum_{i=1}^n{P(AB_i)P(B_i)}$$

Visualize in terms of tree diagram. 

Example 1: For instance suppose there are two electrical components. The chance that the first component fails is 10%. If the first component fails, the chance that the second component fails is 20%. But if the first component works, the chance that the second component fails is 5%. This can be represented as tree

|- first comp works (90%)
       | - second comp works (95%)
       | - second comp fails (5%)
|- first comp fails (10%)
       | - second comp works (80%)
       | - second comp fails (20%)
       
P(at least one of the comp works) = 0.90 * 0.05 + 0.1 * 0.8 = 0.125

Example 2: Suppose two cards are dealth from a well shuffled deck of 52 cards. What is the probability that the second card is black. 


## Odds
* Chance odds: ratio of probabilities, e.g., the following are equivalent P(A) = 3/10; the odds of A are 3 in 10; **the odds in favor of A are 3 to 7**; the odds against A are 7 to 3. 
* Payoff odds: ratio of stakes: $\frac{What~you~get}{what~you~bet}$ (what you get does not include what you bet)
* Fair odds rule: in a fair bet, payoff odds equal chance odds


## Probability of Sequence of Events

$$P(ABC) = P(AB)P(C|AB) = P(A)P(A|B)P(C|AB)$$
or $P(A_1A_2\cdots A_n) = P(A_1)P(A_2|A_1)P(A_3|A_1A_2)\cdots P(A_n|A_1A_2\cdots A_{n-1})$

Example: A contractor is planning a construction project to be completed in three stages. The contractor figures that
1. the chance that the first stage will be completed on time is 0.7
2. given that the first stage is completed on time, the chance that the second stage will be completed on time is 0.8
3. given that both first and second stages are completed on time, the chance that the third state will be completed on time is 0.9

Q1) What is the chance that all the three stages will be completed on time = 0.7 * 0.8 * 0.9 = 0.504. 
Q2) What is the chance that first and second stages are completed on time but not third: $P(ABC^c) = 0.7 * 0.8 * (1 - 0.9) = 0.056$
Q3) Probability that second stage is late ? We cannot compute this as $P(B^c) = P(A)P(B^c|A) + P(A^c)P(B^c|A^c)$ and we don't known $P(B^c|A^c)$


## Probability of Sequence of Events (Eg 2)

A 6 sided die has 1 of its faces painted white and rest 5 its faces painted black. The die is rolled until the first time a white face shows up. What is the chance that this takes three or less rolls. 

Assume: $P(W) = 1/6 = p$ and $P(B) = 5/6 = q$
Now one get white side in the first roll or second roll or third roll. The three situations represents three mutuall exclusive ways that event {white in 3 or less rolls} could happen. Then 

$P(W \leq 3~rolls) = P(W_1) + P(W_2B_1) + P(W_3B_2B_1)$
$ = P(W_1) + P(B_1)P(W_2|B_1) + P(B_1)P(B_2|B_1)P(W_3|B_1B_2)$
$ = P(W_1) + P(B_1)P(W_2) + P(B_1)P(B_2)P(W_3)$ ( since each roll is independent)
$ = p + qp + q^2p = (1 + q + q^2)p = (1 + q + q^2)(1-q) = 1 - q^3$

Question: what is the probability that it takes 4 or more rolls to get a white
Ans: P(4 or more rolls to get white) = 1 - P(white in 3 or less rolls)

## The gambler's rule

Suppose you play a game over and over again, each time with a chance 1/N of winning the game, no matter what the results of previous games. How many times you must play to have a better than 50% chance of at least one win in the k games. 

P(at least one win in k games) = 1 - P(no win in k games)

$ = 1 - (1 - 1/N)^k$

We are looking for the least n such that $ 1 - (1 - 1/N)^k > 1/2$ i.e. $(1-1/N)^k < 1/2$. Assuming -1/N = z and minimizing for k we get
$ (1 + z)^k < 1/2$ 
= $ k log(1+z) < log(1/2)$ 

there is an approximation that says log(1+z) -> z as z approaches zero. 
So we can rewrite above eqn as $kz = log(1) - log(2) = 0 - log(2) = -log(2)$

Since z = -1/N, $-k/N = -log(2)$  i.e. $k = N log(2)$ ~ $k=2/3N$. 


## Binomial Theorem

The sum of probabilities of k success where k ranges from 0 to n is equal to 1 i.e 

$$ \sum_{k=0}^{n}P(k~success~in~n~trials) = \sum_{k=0}^{n}nCk~p^kq^{n-k} = 1$$

## Consecutive Odds for the Binomial Distribution

For independent trials with success probability p, the odds of k successes relative to k-1 success are R(k) to 1 where

$$R(k) = \frac{P(k~success~in~n~trials)}{P(k-1~successes~in~n~trials} = \frac{n-k+1}{k}\frac{p}{q} $$

This simple formula for ration makes it easy to calculate all the probabities in a binomial distribution recursively. 

Example: A pair of fair coins is tossed 8 times. Find the probability of getting both heads on k of these double tosses for k =  0 to 8. 

Solution: The chance of getting both heads on each double toss is 1/2 x 1/2 = 1/4. Thus assuming this as a binomial distribution the success probability is 1/4. Now, we can compute P(k=0) as

$$P(k=0) = 8C0 * (3/4)^8 = 1 * 0.100... = 0.100$$
$$P(k=1) = R(1)P(0) = (8/1)(1/3)(0.100) = 0.267$$
$$P(k=2) = R(2)P(0) = (7/2)(1/3)(0.267) = 0.311$$
and so on


## R2 (R2 squarred error)

		
$$
\begin{equation}
\begin{split}
R^2 & = \frac{\textrm{Explained Squared Error}}{\textrm{Total Squared Error}} \\
	& = 1 - \frac{\textrm{Unexplained Squared Error}}{\textrm{Total Squared Error}} = 1 - \frac{SS_{regression}}{SS_{Total}} = 1 - \frac{\sum_i{({y}_i-\hat{y_i})^2}}{\sum_i{({y}_i-\bar{y_i})^2}} \\
	& \approx 1 - \frac{\textrm{Area Between Given Model and Ideal Model}}{\textrm{Area Between Mean Model And Ideal Model}}
	\end{split}
	\end{equation}
$$

<table>
<td>
<img src="./images/r2_area.png" width=300 />
</td>
<td>
<img src="./images/r-squared.png" width=300 />
</td>
</table>

**Explanation 1**: It is the opportunity area, as defined by the mean model, that is covered by the regression model. The right top corner defines the ideal model for which 100% of samples have 0 zero. The mean model defines the best model we can have without any predictor. The green region is the improvement the regression model made over the mean model.

**Explanation 2** It is the % of explained error of total area. Total error is the defined as the squared error between actual value and mean value (mean model). Some of that error will go away with our regression model and is referred as expalined error. The error between our model and actual value still remains unexplained. 

Theoretically, $R^2$ can range from -1 to 1. It will be < 0 when the fittted model is worse than mean model. In praticse, it is often considered to range between 0 and 1 .


## Linear Regression and Correlation

For a single variable, linear Regression can be represented as $y = a + bx$ where 
$$b = r \frac{S_y}{S_x}$$
and $$r = \textrm{pearson correlation coefficient} = \frac{\sqrt{(x-\bar{x})(y-\bar{y})}}{(n-1)S_xS_y}$$

Combining above two questions, slope of linear regression is given as

$$b = \frac{\sqrt{(x-\bar{x})(y-\bar{y})}}{nS_x} = \frac{\sqrt{(x-\bar{x})(y-\bar{y})}}{\sum{(x-\bar{x})^2}}$$
 
 and a is given as $a = \bar{y} - b\bar{x}$
 
 
## Modeling exponential target using linear regression
 
 If the model is of tbe form $y = e^{a+bx}$ then we can take log (base to e) and coonvert into $ln(y) = a + bx$ 

* Note that y has to be strictly positive as for $Y <=0$, $ln(y)$ is undefined. 
 	

## Interaction Variables in Regression

Example: Model Height of plant (H) as a function of amount of bacteria (B) in soil and whether the plant is located in partial or full sun (S). If plant is in partial sun then S = 0	and if in full sun then S = 1. 

The optimal equation with interaction term is 

$$Height = B0 + B1*Bacteria + B2*Sun + B3 \* Bacteria \* Sun$$

**Question: What is the unique effect of Bacteria.**

Due to the interaction term, the effect of Bacteria on Height is different for different values of Sun.  So the unique effect of Bacteria on Height is not limited to B1 but also depends on the values of B3 and Sun. The unique effect of Bacteria is represented by everything that is multiplied by Bacteria in the model: B1 + B3*Sun. B1 is now interpreted as the unique effect of Bacteria on Height only when Sun = 0.

Another way of saying this is that the slopes of the regression lines between height and bacteria count are different for the different categories of sun. B3 indicates how different those slopes are.

**Question: How to interpret effect of B2 and Sun**

Interpreting B2 is more difficult. B2 is the effect of Sun when Bacteria = 0. Since Bacteria is a continuous variable, it is unlikely that it equals 0 often, if ever, so B2 can be virtually meaningless by itself.

Instead, it is more useful to understand the effect of Sun, but again, this can be difficult. The effect of Sun is B2 + B3*Bacteria, which is different at every one of the infinite values of Bacteria. For that reason, often the only way to get an intuitive understanding of the effect of Sun is to plug a few values of Bacteria into the equation to see how Height, the response variable, changes.

## How to do multiple regression (intutively)

* Step 1: remove x1 from y to get y1 i.e learn $y = \lambda_1 x_1 + y_1$
* Step 2: remove x1 from x2 to get x2 indepndent of x1 $x_{2,1}$: $x_2 = \lambda_2 x_1 + x_{2,1}$
* Step 3: remove x1 from x3 to get x3 independent of x1 $(x_{3,1})$: $x_3 = \lambda_3 x_1 + x_{3,1}$
* Step 4: remove x_2,1 from y_1 to get $y_{2,1}$ i.e $y_1 = \lambda_4 x_{2,1} + y_{2,1}$


## Adjusted R2

One of the problems with R2 is that assumes all variables explains the variation in the dependent variable. However, it might be possible that only k of n features are actually meaningful and help explain variation in the dependent variable. As a result, R2 only increases with adding more and more predictors. 

The adjusted R-squared compensates for the addition of variables and only increases if the new predictor enhances the model above what would be obtained by probability. Conversely, it will decrease when a predictor improves the model less than what is predicted by chance. It is given as

$$\bar{R^2} = 1 - \frac{Var(e_i)/(n-k-1)}{Var(y)/(n-1)} = 1 - \frac{Var(e)}{Var(y)} \times \frac{n-1}{n-k-1} = 1 - \frac{(1-R^2)(n-1)}{n-k-1}$$

## Feature Engineering Tips

1. Missing Values:
	1. If numeric: replace with median and also add another column with binary value. Use value 1 to indicate that the original value was missing otherwise 0
	2. If categorical column, 


## Linear Regression Summary

**Ordinary Least Square**

* Hypothesis Function: $h_\theta=\sum_{i}^n\theta_ix_i$
* Cost Function: $J_\theta=\frac{1}{2m}\sum_{i}^m\left(y^i - h_\theta(x^i)\right)^2$
* Derivative: $\frac{dJ_\theta}{d\theta_i} = \frac{-1}{m}\sum_{i}^m(y^i-h_\theta(x^i))x^i_j$ (**don't for negative sign**)
* Update: $\theta_i = \theta_i - \eta\frac{dJ_\theta}{d\theta_i}$ (**note all theta are updated together**)

**L1 Regularization (Lasso Regression)**

* Cost Function: $J_\theta=\frac{1}{2m}\sum_{i}^m\left(y^i - h_\theta(x^i)\right)^2 + \lambda\sum_i^n|\theta_i|$
* Derivative: $\frac{dJ_\theta}{d\theta_i} = \frac{-1}{m}\sum_{i}^m(y^i-h_\theta(x^i))x^i_j + \lambda \frac{\theta_i}{\sqrt{\theta^2}}$ 
* Note that $\frac{\theta_i}{\sqrt{\theta^2}}$ is +1 if $\theta_i$ is positive or -1 if $\theta_i$ is negative. Also it forces the parameters to zero because for any other value $\neq 0$ of $\theta_i$ we apply $\lambda$ as penalty. 
* It does not have a closed form solution as 


**L2 Regularization (Ridge Regression)**

* Cost Function: $J_\theta=\frac{1}{2m}\sum_{i}^m\left(y^i - h_\theta(x^i)\right)^2 + \lambda\sum_i^n|\theta_i|^2$
* Derivative: $\frac{dJ_\theta}{d\theta_i} = \frac{-1}{m}\sum_{i}^m(y^i-h_\theta(x^i))x^i + 2\lambda\theta_i$
* It has a closed form solution 


**Why L1 leads to sparse model:**: Think what happens in $dJ(\theta)/d\theta_i$ as $\theta_i$ approaches to zero. In the case of L1 regularization, the L1 penalty remains same($\lambda$) unless $\theta_i$ becomes zero. The only way to minimize penalty is to froce $\theta_i$ to be zero. However, in case of L2 regularization, the penalty ($2\lambda\theta_i$) approaches zero as $\theta_i$ approaches zero. 
In contrast to L1 regularization, as $\theta_i$ approaches zero, the penalty becomes smaller and samller and eventually approaches zero. 



## Derivative of Absolute Value

$$y=|x| = \sqrt{x^2} = (x^2)^{1/2}$$

$
\begin{equation}
\begin{split}
\frac{dy}{dx} = \frac{1}{2}(x^2)^{1/2-1} (2x) \\
& = x(x^2)^{-1/2}\\
& = \frac{x}{\sqrt{x^2}}\\
	\end{split}
	\end{equation}
	$
	
## Basis Function

The transformations done on input variables are referred as basis function. 


## Dimensionality Reduction Techniques

**Pratical Approaches**

* Missing Value Ratio: Remove columns with significant portion of missing values
* High Correlation Filter: Features exhibiting high degree of correlation can be removed. Calculate pearson correlation coefficient between variables and use ward clustering to identify to identify correated variables at different depths. 
* Lasso Regression (L1 Regularization)
* PCA

Source: A Survey of dimensionality reduction techniques

**Methods Based On Statistical and Information Theory**

* Vector Quantification and Mixture Models: Reduce to 1 dimension by assigning all instances to different class using techniques such as k-means. 
* Principal Component Analysis (PCA): Use PCA to find fewer dimensions that captures most of the variance. 
* Principal curves, surfaces and manifolds: PCA is good when data follow some **linear** manifold. However sometimes data follow curved surfaces and using PCA forces onto linear manifold. 
* Generative Topograhic Mapping:
* Self Organizing Maps (SOM): 
* Independent Component Analysis (ICA)
* Elastic maps, nets, principal graphs and princiapl trees
* Kernel PCA and Multidimensional Scaling
* Kernel Entropy Component Analysis
* Robust PCA
* Factor Analysis (FA)

**Methods Based on Dictionaries**

1. Non negative Matrix Factorization
2. Principal Tensor Analysis and Non Negative Tensor Factorization
3. Generalized SVD
3. Sparse Representation and overcomplete dictionaries
4. 


## Adaboost
* Proposed by Yova Freund and Robert Shapire in 1995
* Iteratively build model where in each iteration the weight of correctly classified example is decreased and vice-versa. The sample weight is calculated as follows:
$$w_i^{m+1} = \frac{w_i^{m} e^{-\theta_m}}{Z_{m+1}}$$

where $Z_{m+1}$ is the normalization factor that is sum of numerator i.e. $\sum_i^n{w_i^me^{-\theta_m}}$. $w_i^m$ is the instance weight for the m th classifier and $\theta_m$ is the weight of the mth classifier itself. Classifier weight $\theta_m$ is given as:

$\theta_m = \frac{1}{2}ln(\frac{1-E_m}{E_m})$

$E$ is total error and is the sum of weight of incorrectly classified examples and is given as 

$$E = \sum_{i=1}^{n}w_iI(y_i \neq h_j(x_i))$$


## Linear Spline
* Used in the paper: "the case for learned index structure" and "learned sorted". Instead of linear regression using MSE Loss , linear regression usin Spline fitting is cheaper to compute ang gives better monotonicity. 
* The idea of regression spline is to split the data into multiple buckets and fit each bin with a separate model. -- This helps 


## Logistic Regression & Binary Cross Entropy
Logistic regression uses binary cross entropy as the cost function. Entropy of a system is calculated as follows: 
$$H(y,\hat{y}) = -\sum_{i=1}^{n}P(y_i)log(P(\hat{y}_i))$$.

For a binary system, the entropy calculation can be further written as 

$$H(y,\hat{y}) = -\sum_{i=1}^{n}\left[P(y_i)log(P(\hat{y}_i)) + (1-y_i)log(P(1-\hat{y}_i))\right]$$.

## Vanishing Gradient in DL
Gradient becoming zero is known as vanishing gradient problem. It can happen for multiple reasons, such as:
1. **bad activation function**: When using sigmoid activation function, the output value ranges between 0 and 1. Since in most cases the value is less than 1, when computing backpropagation the derivative value will tend to be zero and thereby the parameters won't get updated. TanH address this problem by having a range of -1 to 1 but still suffer from vanishing gradient problem. Relu uses a threshold function and the value ranges from 0 to infinity and hence is able to handle vanishing gradient problem in a much better way. 
2. **bad initialization value**: If the random initialization weights are too small or too large, it leads to vanishing gradient. The range of the uniform distribution should be proportional to number of neurons in a layer. There are different techniques used to initialize weights. For instance Pytorch initializes weights by uniformly selecting weights between $-1/\sqrt{L_i}$ and $+1/\sqrt{L_i}$ where L is the number of neurons in a layer. 

## Weight Initialization Techniques In DL
While we randomly initialize weights using uniform distribution, different techniques propose different range for the uniform distribution. Below are different ranges proposed by different techniques: 

1. PyTorch Default Method: $-1/\sqrt{L_i}$ and $+1/\sqrt{L_i}$ where L is the number of neurons in a layer. 
2. **Xavier Method**: $-\sqrt{6}/\sqrt{L_{out}-L{in}}$ and $\sqrt{6}/\sqrt{L_{out}-L{in}}$, where $L_{in}$ is the number neurons in the input layer and $L_{out}$ is the number of neurons in the output layer.
3. **He Method**: 


## Why x / 0 is undefined
Anynumber divided by zero is actually infinity. For instance if we 5/1 will be 5, 5/0.5 will be 10, 5/0.01 = 500 and so forth. As you can see as we approach zero, the quotient becomes increasingly large and approaches infinity. However, this means any number divided by zero will be infinity, or in other words, infinity multiplied by zero can be any number. This is meaningless in mathematics. 

Here is another way to show how allowing divison by zero leads to absurd results

* Assuming $0 * 1 = 0; 0 * 2 = 0$
* Then $ 0 * 1 = 0 * 2$
* Dividing both sides by zero: $\frac{0 * 1}{0} = \frac{0 * 2}{0}$
* yields: $1 = 2$ ..

This is the reason that 0/0 is also undefined.  

## Zeno's paradox
1. **Dichotomy Paradox**: "That which is in locomotion must arrive at the half-way stage before it arrives at the goal" (Aristotle, Physics). Support a man wishes to take a step. Before he can take a full step, he must take a half step. Before he can take a half step, he must take a quarter step and so forth. Then how to take the first step. 
2. **Achilles and the tortoise**: In the paradox of Achilles and the tortoise, Achilles is in a footrace with the tortoise. Achilles allows the tortoise a head start of 100 meters, for example. Suppose that each racer starts running at some constant speed, one faster than the other. After some finite time, Achilles will have run 100 meters, bringing him to the tortoise's starting point. During this time, the tortoise has run a much shorter distance, say 2 meters. It will then take Achilles some further time to run that distance, by which time the tortoise will have advanced farther; and then more time still to reach this third point, while the tortoise moves ahead. Thus, whenever Achilles arrives somewhere the tortoise has been, he still has some distance to go before he can even reach the tortoise. As Aristotle noted, this argument is similar to the Dichotomy.
3. **Arrow Paradox:** In the arrow paradox, Zeno states that for motion to occur, an object must change the position which it occupies. He gives an example of an arrow in flight. He states that in any one (duration-less) instant of time, the arrow is neither moving to where it is, nor to where it is not. It cannot move to where it is not, because no time elapses for it to move there; it cannot move to where it is, because it is already there. In other words, at every instant of time there is no motion occurring. If everything is motionless at every instant, and time is entirely composed of instants, then motion is impossible.

## Laplacian Smoothing
* also known as additive smoothing or lidstone smoothing
* it is a technique to smooth probabilities of categorical data. When applying bayesian statistics on NLP, the probability of an unknown word would be zero and would cause the whole probability to be zero. Instead we replace the probability of the word with a very small probability and take the same amount from rest of the words. This process is known as laplacing smoothing. 

$$ P(w_i | class) = \frac{freq(w_i, class) + \alpha}{N_class + \alpha d} $$
where
* $\alpha$ is the default weight of a word (usually 1)
* d is the number of unique words in the vocabulary. 
* $N_class$ is the total weight of all the words associated with the given class. 

## Cosine Similarity
$$ cos(\beta) = \frac{v \dot w}{||v||||w||} $$

```
cosine_similarity = np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w)) 
```

* In NLP, consine angle is used to determine similarity between two documents. 
* The range of consine similarity is -1 and 1. While 1 indicates exactly similar documents, -1 indicates that documents are exactly opposite. Instead of consine similarity, it might be easier to   convert it into **cosine distance** which is given as $cos\_dist = 1 - cos\_similarity$. The range for $cos\_dist$ will be 0 to 2. Large the value, the more different the two vectors are. 


## Covariance and Covariance Matrix
Covariance can be generally described as average relationship between coordinates of points. For instance we have a 2D data with three points: (-2, 1), (0, 0) and (2, -1). Covariance of this dataset is given as : $\frac{(-2 * 1) + (0 * 0) + (2 * -1)}{3} = \frac{-2 + 0 - 2}{3} = \frac{-4}{3}$. 


## Frobenius Norm 
$$||A||_F = \sqrt{tr(A^TA)} = \sqrt{\sum_{i=1}^{m}\sum_{j=1}^n|a_{ij}|^2}$$

Example: if A is 2 X 2 Matrix such as 
A = [[2, 2], [2, 2]], then Frobenius norm will be $\sqrt{2^2 + 2^2 + 2^2 + 2^2} = 4$  

* Frobenius norm is differentiable and hence can be used with Gradient optimization techniques
* Also known as euclidean norm. Note that usually euclidean norm is applied on a single vector whereas Frobenius norm is applied on a matrix.


## Locality Sensitive Hashing
* It is a technique that hashes similar input items into the same buckets with high probability. It uses multiple hashing algorithms to repeateadly hash a single data point into multiple buckets 
* Implementation (for N dimensional data): Assuming if you have an N dimensional data (such as in the case of NLP) and you want to hash word vectors that are you close to each other. In this case LSH can be used as follows:
	* Randomly initialize a normal vector. A normal vector uniquely identifies a hyperplane that splits the vector space into exactly two sub regions. Compute direction of all the word vectors relative to this hyperplane.  The sign of the dot product between the normal vector and the candidate work indicates on which side of the hyperplane is the word. If the magnitude of the dot product is <=0 then set to zero. 
	* Take the sum of all the directions using the following formula: $\sum_i{2^i h^i}$ where i is the ith randomnly initialized vector and $h^i$ is the sign of mangitude of a word vector relative to the ith normal vector i.e. if the dot product is -10, then h = 0. If it is 10 then h = 1. 
	* Repeat the above process to generate many different hashes. 

## Machine Translation Problem
Problem: Convert English to French
1. Generate word vectors for English and French
2. The aim is to learn rotation matrix (R) that transforms english word vector space into french word vector space and minimizes the difference between translation i.e. $argmin || XR - Y ||$ where X is the matrix of word embedding of english words and Y is the matrix of word embedding of french words. R is the rotation matrix that we. want to learn. 
3. Use mean squared Frobenius as the loss function. Assume $A = XR - Y$, Frobenius norm is defined as 

$$ ||A||_F = \sqrt{\sum_{i=1}^m\sum_{j=1}^n|a_{ij}|^2} $$

Hence the actual loss function is $\frac{1}{m}||XR-Y||_F^2$

4. The above loss function is differentiable and hence we can use gradient descent approach to identify R matrix. The gradient of $\frac{1}{m}||XR-Y||_F^2$ with respect to R is $\frac{2}{m}X^T(XR-Y)$. 
5. Once we have the R matrix, we can use dot product of any new english word and identity all the closet french words to XR vector. 

## Emission Probabilities

It is used in reference to Markov chains (POS tagging). Emission probabilities is the joint probability of a term and POS tag. More technically, it is the probability of going from a hidden state (POS tag) to the observable state (term). For instance, emission probability represents probability of going from verb state to term say "eat". 

## Viterbi Algorithm
It is a graph algorithm to find shortest/longest path. It is used in parts of speech tagging to determine most likely sequence of hidden states in the context of hidden markov model (HMM). The algorithm involves forward and backward or (traceback) pass. In the forward pass you move from starting point to the next possible nodes and keep track of accumulated distance or probabilities in a matrix, say C. Also we retain the source in another matrix, say D. Once we have completed forward pass, we do a backward pass to find the smallest/largest route. 

Source: https://www.youtube.com/watch?v=6JVqutwtzmo


## Law of 72 (Investment)
The amount of time it takes for your principal to double at X% return is approximately equal to 72 / X. For instance, if you investment returns 7% return then it will take $72 / 7 \approx 10.5$ years to double

Caveats:

* applies to compound interest
* 72 applies within the range of 6% to 10% return. Outside of the range, adjust 72 using this formula: $72 - (8 - r)/3$. For instance assume rate of return is 2%. Since 2% is less than 6%, adjust 72 to be $72 - (8 - 2)/3 = 70$. Then the amount of time for the investment to double will be $70 / 2 = 35$ years. For 11% return, it will be 
$$\frac{(72 + 3/3)}{11} \approx \frac{73}{11} \approx 6.6~years$$

