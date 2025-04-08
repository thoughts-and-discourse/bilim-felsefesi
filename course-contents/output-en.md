# Notation

It is very difficult to come up with a single, consistent notation to
cover the wide variety of data, models and algorithms that we discuss in
this book. Furthermore, conventions differ between different fields
(such as machine learning, statistics and optimization), and between
different books and papers within the same field. Nevertheless, we have
tried to be as consistent as possible. Below we summarize most of the
notation used in this book, although individual sections may introduce
new notation. Note also that the same symbol may have different meanings
depending on the context, although we try to avoid this where possible.

## Common mathematical symbols

We list some common symbols below.

::: center
  Symbol                   Meaning
  ------------------------ -------------------------------------------------------------
  $\infty$                 Infinity
  $\rightarrow$            Tends towards, e.g., $n \rightarrow \infty$
  $\propto$                Proportional to, so $y=a x$ can be written as $y \propto x$
  $\triangleq$             Defined as
  $O(\cdot)$               Big-O: roughly means order of magnitude
  $\mathbb{Z}_{+}$         The positive integers
  $\mathbb{R}$             The real numbers
  $\mathbb{R}_{+}$         The positive reals
  $\mathcal{S}_{K}$        The $K$-dimensional probability simplex
  $\mathcal{S}_{++}^{D}$   Cone of positive definite $D \times D$ matrices
  $\approx$                Approximately equal to
  $\{1, \ldots, N\}$       The finite set $\{1,2, \ldots, N\}$
  $1: N$                   The finite set $\{1,2, \ldots, N\}$
  $[\ell, u]$              The continuous interval $\{\ell \leq x \leq u\}$.
                           
:::

## Functions

Generic functions will be denoted by $f$ (and sometimes $g$ or $h$ ). We
will encounter many named functions, such as $\tanh (x)$ or $\sigma(x)$.
A scalar function applied to a vector is assumed to be applied
elementwise, e.g.,
$\boldsymbol{x}^{2}=\left[x_{1}^{2}, \ldots, x_{D}^{2}\right]$.
Functionals (functions of a function) are written using \"blackboard\"
font, e.g., $\mathbb{H}(p)$ for the entropy of a distribution $p$. A
function parameterized by fixed parameters $\boldsymbol{\theta}$ will be
denoted by $f(\boldsymbol{x} ; \boldsymbol{\theta})$ or sometimes
$f_{\boldsymbol{\theta}}(\boldsymbol{x})$. We list some common functions
(with no free parameters) below.

## Common functions of one argument

::: center
  Symbol               Meaning
  -------------------- ------------------------------------------------------------------------------
  $\lfloor x\rfloor$   Floor of $x$, i.e., round down to nearest integer
  $\lceil x\rceil$     Ceiling of $x$, i.e., round up to nearest integer
  $\neg a$             logical NOT
  $\mathbb{I}(x)$      Indicator function, $\mathbb{I}(x)=1$ if $x$ is true, else $\mathbb{I}(x)=0$
  $\delta(x)$          Dirac delta function, $\delta(x)=\infty$ if $x=0$, else $\delta(x)=0$
  $|x|$                Absolute value
  $|\mathcal{S}|$      Size (cardinality) of a set
  $n !$                Factorial function
  $\log (x)$           Natural logarithm of $x$
  $\exp (x)$           Exponential function $e^{x}$
  $\Gamma(x)$          Gamma function, $\Gamma(x)=\int_{0}^{\infty} u^{x-1} e^{-u} d u$
  $\Psi(x)$            Digamma function, $\Psi(x)=\frac{d}{d x} \log \Gamma(x)$
  $\sigma(x)$          Sigmoid (logistic) function, $\frac{1}{1+e^{-x}}$
:::

## Common functions of two arguments

::: center
  Symbol                                        Meaning
  --------------------------------------------- ------------------------------------------------------------------
  $a \wedge b$                                  logical AND
  $a \vee b$                                    logical OR
  $B(a, b)$                                     Beta function, $B(a, b)=\frac{\Gamma(a) \Gamma(b)}{\Gamma(a+b)}$
  $n \choose k$                                 $n$ choose $k$, equal to $n ! /(k !(n-k) !)$
  $\delta_{i j}$                                Kronecker delta, equals $\mathbb{I}(i=j)$
  $\boldsymbol{u} \odot \boldsymbol{v}$         Elementwise product of two vectors
  $\boldsymbol{u} \circledast \boldsymbol{v}$   Convolution of two vectors
:::

## Common functions of $>2$ arguments

::: center
  Symbol                     Meaning
  -------------------------- ------------------------------------------------------------------------------------------------------------
  $B(\boldsymbol{x})$        Multivariate beta function, $\frac{\prod_{k} \Gamma\left(x_{k}\right)}{\Gamma\left(\sum_{k} x_{k}\right)}$
  $\Gamma(\boldsymbol{x})$   Multi. gamma function, $\pi^{D(D-1) / 4} \prod_{d=1}^{D} \Gamma(x+(1-d) / 2)$
:::

Draft of \"Probabilistic Machine Learning: An Introduction\". May 9,
2022

$$\mathcal{S}(\boldsymbol{x}) \quad \text { Softmax function, }\left[\frac{e^{x_{c}}}{\sum_{c^{\prime}=1}^{C} e^{x^{\prime}}}\right]_{c=1}^{C}$$

## Linear algebra

In this section, we summarize the notation we use for linear algebra
(see Chapter 7 for details).

## General notation

Vectors are bold lower case letters such as
$\boldsymbol{x}, \boldsymbol{w}$. Matrices are bold upper case letters,
such as $\mathbf{X}$, W. Scalars are non-bold lower case. When creating
a vector from a list of $N$ scalars, we write
$\boldsymbol{x}=\left[x_{1}, \ldots, x_{N}\right]$; this may be a column
vector or a row vector, depending on the context. (Vectors are assumed
to be column vectors, unless noted otherwise.) When creating an
$M \times N$ matrix from a list of vectors, we write
$\mathbf{X}=\left[\boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{N}\right]$
if we stack along the columns, or
$\mathbf{X}=\left[\boldsymbol{x}_{1} ; \ldots ; \boldsymbol{x}_{M}\right]$
if we stack along the rows.

## Vectors

Here is some standard notation for vectors. (We assume $\boldsymbol{u}$
and $\boldsymbol{v}$ are both $N$-dimensional vectors.)

::: center
  Symbol                                        Meaning
  --------------------------------------------- ----------------------------------------------------------------------
  $\boldsymbol{u}^{\top} \boldsymbol{v}$        Inner (scalar) product, $\sum_{i=1}^{N} u_{i} v_{i}$
  $\boldsymbol{u} \boldsymbol{v}^{\top}$        Outer product $(N \times N$ matrix)
  $\boldsymbol{u} \odot \boldsymbol{v}$         Elementwise product, $\left[u_{1} v_{1}, \ldots, u_{N} v_{N}\right]$
  $\boldsymbol{v}^{\top}$                       Transpose of $\boldsymbol{v}$
  $\operatorname{dim}(\boldsymbol{v})$          Dimensionality of $\boldsymbol{v}$ (namely $N$ )
  $\operatorname{diag}(\boldsymbol{v})$         Diagonal $N \times N$ matrix made from vector $\boldsymbol{v}$
  $\mathbf{1}$ or $\mathbf{1}_{N}$              Vector of ones (of length $N$ )
  $\mathbf{0}$ or $\mathbf{0}_{N}$              Vector of zeros (of length $N$ )
  $\|\boldsymbol{v}\|=\|\boldsymbol{v}\|_{2}$   Euclidean or $\ell_{2}$ norm $\sqrt{\sum_{i=1}^{N} v_{i}^{2}}$
  $\|\boldsymbol{v}\|_{1}$                      $\ell_{1}$ norm $\sum_{i=1}^{N}\left|v_{i}\right|$
:::

## Matrices

check what Here is some standard notation for matrices. (We assume
$\mathbf{S}$ is a square $N \times N$ matrix, $\mathbf{X}$ and
$\mathbf{Y}$ are of size $M \times N$, and $\mathbf{Z}$ is of size
$M^{\prime} \times N^{\prime}$.)

::: center
  Symbol                             Meaning
  ---------------------------------- -----------------------------------------------------
  $\mathbf{X}_{:, j}$                $j^{\prime}$ th column of matrix
  $\mathbf{X}_{i,:}$                 $i$ 'th row of matrix (treated as a column vector)
  $X_{i j}$                          Element $(i, j)$ of matrix
  $\mathbf{S} \succ 0$               True iff $\mathbf{S}$ is a positive definite matrix
  $\operatorname{tr}(\mathbf{S})$    Trace of a square matrix
  $\operatorname{det}(\mathbf{S})$   Determinant of a square matrix
  $|\mathbf{S}|$                     Determinant of a square matrix
:::

::: center
  ----------------------------------- ----------------------------------------------
  $\mathbf{S}^{-1}$                   Inverse of a square matrix
  $\mathbf{X}^{\dagger}$              Pseudo-inverse of a matrix
  $\mathbf{X}^{\top}$                 Transpose of a matrix
  $\operatorname{diag}(\mathbf{S})$   Diagonal vector extracted from square matrix
  $\mathbf{I}$ or $\mathbf{I}_{N}$    Identity matrix of size $N \times N$
  $\mathbf{X} \odot \mathbf{Y}$       Elementwise product
  $\mathbf{X} \otimes \mathbf{Z}$     Kronecker product (see Section 7.2.5)
  ----------------------------------- ----------------------------------------------
:::

## Matrix calculus

In this section, we summarize the notation we use for matrix calculus
(see Section 7.8 for details).

Let $\boldsymbol{\theta} \in \mathbb{R}^{N}$ be a vector and
$f: \mathbb{R}^{N} \rightarrow \mathbb{R}$ be a scalar valued function.
The derivative of $f$ wrt its argument is denoted by the following:

$$\nabla_{\boldsymbol{\theta}} f(\boldsymbol{\theta}) \triangleq \nabla f(\boldsymbol{\theta}) \triangleq \nabla f \triangleq\left(\begin{array}{ccc}
        \frac{\partial f}{\partial \theta_{1}} & \cdots & \frac{\partial f}{\partial \theta_{N}}
    \end{array}\right)$$

The gradient is a vector that must be evaluated at a point in space. To
emphasize this, we will sometimes write

$$\left.\boldsymbol{g}_{t} \triangleq \boldsymbol{g}\left(\boldsymbol{\theta}_{t}\right) \triangleq \nabla f(\boldsymbol{\theta})\right|_{\boldsymbol{\theta}_{t}}$$

We can also compute the (symmetric) $N \times N$ matrix of second
partial derivatives, known as the Hessian:

$$\nabla^{2} f \triangleq\left(\begin{array}{ccc}
        \frac{\partial^{2} f}{\partial \theta_{1}^{2}} & \cdots & \frac{\partial^{2} f}{\partial \theta_{1} \partial \theta_{N}} \\
        & \vdots & \\
        \frac{\partial^{2} f}{\partial \theta_{N} \theta_{1}} & \cdots & \frac{\partial^{2} f}{\partial \theta_{N}^{2}}
    \end{array}\right)$$

The Hessian is a matrix that must be evaluated at a point in space. To
emphasize this, we will sometimes write

$$\left.\mathbf{H}_{t} \triangleq \mathbf{H}\left(\boldsymbol{\theta}_{t}\right) \triangleq \nabla^{2} f(\boldsymbol{\theta})\right|_{\boldsymbol{\theta}_{t}}$$

## Optimization

In this section, we summarize the notation we use for optimization (see
Chapter 8 for details).

We will often write an objective or cost function that we wish to
minimize as $\mathcal{L}(\boldsymbol{\theta})$, where
$\boldsymbol{\theta}$ are the variables to be optimized (often thought
of as parameters of a statistical model). We denote the parameter value
that achieves the minimum as
$\boldsymbol{\theta}_{*}=\operatorname{argmin}_{\boldsymbol{\theta} \in \Theta} \mathcal{L}(\boldsymbol{\theta})$,
where $\Theta$ is the set we are optimizing over. (Note that there may
be more than one such optimal value, so we should really write
$\boldsymbol{\theta}_{*} \in \operatorname{argmin}_{\boldsymbol{\theta} \in \Theta} \mathcal{L}(\boldsymbol{\theta})$.)

When performing iterative optimization, we use $t$ to index the
iteration number. We use $\eta$ as a step size (learning rate)
parameter. Thus we can write the gradient descent algorithm (explained
in Section 8.4) as follows:
$\boldsymbol{\theta}_{t+1}=\boldsymbol{\theta}_{t}-\eta_{t} \boldsymbol{g}_{t}$.

Draft of \"Probabilistic Machine Learning: An Introduction\". May 9,
2022 We often use a hat symbol to denote an estimate or prediction
(e.g., $\hat{\boldsymbol{\theta}}, \hat{y})$, a star subscript or
superscript to denote a true (but usually unknown) value (e.g.,
$\boldsymbol{\theta}_{*}$ or $\boldsymbol{\theta}^{*}$ ), an overline to
denote a mean value (e.g., $\overline{\boldsymbol{\theta}}$ ).

## Probability

In this section, we summarize the notation we use for probability theory
(see Chapter 2 for details).

We denote a probability density function (pdf) or probability mass
function (pmf) by $p$, a cumulative distribution function (cdf) by $P$,
and the probability of a binary event by $\operatorname{Pr}$. We write
$p(X)$ for the distribution for random variable $X$, and $p(Y)$ for the
distribution for random variable $Y$ - these refer to different
distributions, even though we use the same $p$ symbol in both cases. (In
cases where confusion may arise, we write $p_{X}(\cdot)$ and
$p_{Y}(\cdot)$.) Approximations to a distribution $p$ will often be
represented by $q$, or sometimes $\hat{p}$.

In some cases, we distinguish between a random variable (rv) and the
values it can take on. In this case, we denote the variable in upper
case (e.g., $X$ ), and its value in lower case (e.g., $x$ ). However, we
often ignore this distinction between variables and values. For example,
we sometimes write $p(x)$ to denote either the scalar value (the
distribution evaluated at a point) or the distribution itself, depending
on whether $X$ is observed or not.

We write $X \sim p$ to denote that $X$ is distributed according to
distribution $p$. We write $X \perp Y \mid Z$ to denote that $X$ is
conditionally independent of $Y$ given $Z$. If $X \sim p$, we denote the
expected value of $f(X)$ using

$$\mathbb{E}[f(X)]=\mathbb{E}_{p(X)}[f(X)]=\mathbb{E}_{X}[f(X)]=\int_{x} f(x) p(x) d x$$

If $f$ is the identity function, we write
$\bar{X} \triangleq \mathbb{E}[X]$. Similarly, the variance is denoted
by

$$\mathbb{V}[f(X)]=\mathbb{V}_{p(X)}[f(X)]=\mathbb{V}_{X}[f(X)]=\int_{x}(f(x)-\mathbb{E}[f(X)])^{2} p(x) d x$$

If $\boldsymbol{x}$ is a random vector, the covariance matrix is denoted

$$\operatorname{Cov}[\boldsymbol{x}]=\mathbb{E}\left[(\boldsymbol{x}-\overline{\boldsymbol{x}})(\boldsymbol{x}-\overline{\boldsymbol{x}})^{\top}\right]$$

If $X \sim p$, the mode of a distribution is denoted by

$$\hat{x}=\operatorname{mode}[p]=\underset{x}{\operatorname{argmax}} p(x)$$

We denote parametric distributions using
$p(\boldsymbol{x} \mid \boldsymbol{\theta})$, where $\boldsymbol{x}$ are
the random variables, $\boldsymbol{\theta}$ are the parameters and $p$
is a pdf or pmf. For example,
$\mathcal{N}\left(x \mid \mu, \sigma^{2}\right)$ is a Gaussian (normal)
distribution with mean $\mu$ and standard deviation $\sigma$.

## Information theory

In this section, we summarize the notation we use for information theory
(see Chapter 6 for details).

If $X \sim p$, we denote the (differential) entropy of the distribution
by $\mathbb{H}(X)$ or $\mathbb{H}(p)$. If $Y \sim q$, we denote the KL
divergence from distribution $p$ to $q$ by $D_{\mathbb{K K} L}(p \| q)$.
If $(X, Y) \sim p$, we denote the mutual information between $X$ and $Y$
by $\mathbb{I}(X ; Y)$.
