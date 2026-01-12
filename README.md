# K-Means Clustering

## Intuition

K-Means clustering is an unsupervised machine learning algorithm. It is founded in statistics and computing distance from cluster centroids. It is used for unlabeled data. It used for finding trends in the data. The Elbow Method in K-Means — is a standard way to pick the “best” number of clusters K by plotting the within-cluster sum of squares (WCSS).

## Description of Algorithm in $\mathbb{R}^d$

The $d$-dimensional Cartesian product of sets $X_1, X_2, \dots, X_d$ is:

$$
X_1 \times X_2 \times \cdots \times X_d = \{ (x_1, x_2, \dots, x_d) \mid x_i \in X_i,\, i=1,\dots,d \}
$$

If all sets are the same, $X^d = X \times \dots \times X$ ($d$ times):

$$
X^d = \{ (x_1, x_2, \dots, x_d) \mid x_i \in X,\, i=1,\dots,d \}
$$

For real-valued vectors:

$$
\mathbb{R}^d = \underbrace{\mathbb{R} \times \mathbb{R} \times \cdots \times \mathbb{R}}_{d \text{ times}}
= \{ (x_1, x_2, \dots, x_d) \mid x_i \in \mathbb{R},\, i=1,\dots,d \}
$$

For practical applications each $X_i \in X^d,\, i=1,\dots,d$, $X_d \subseteq \mathbb{R}$. Hence $X^d \subseteq \mathbb{R}^d$

That is we would be operating on a bounded hypercube in d-dimensions.
## How this would look in $\mathbb{R}^2$
In $\mathbb{R}^2$ we could write this as,
$$
X \times Y = \{ (x, y) \mid x \in X,\, y \in Y\}
$$
or since X and Y if real would be,
$$
X \times Y = \{ (x_i, y_i) \mid x_i \in X,\, y_i \in Y\}\subseteq \mathbb{R}^2$$

Here the dimension is $d=2$ and the value i goes from 1 to $n$, the number of data points. Let's proceed with the algorithm in full generality, that is $\mathbb{R}^d$. Notice $x$ is a vector below and X is a matrix. That is $x\in\mathbb{R}^d$ and $X \in \mathbb{R}^{n \times d} \quad \text{(n rows, d columns)}$

# Algorithm: K-Means Clustering

**Input:** 
- Dataset $X = \{x_1, x_2, \dots, x_n\}$ with $x_i \in \mathbb{R}^d$  
- Number of clusters $K$

**Output:** 
- Cluster assignments $C = \{c_1, \dots, c_n\}$  
- Cluster centroids $\{\mu_1, \dots, \mu_K\}$

---

**Steps:**

1. **Initialize** $K$ cluster centroids $\mu_1, \dots, \mu_K$ randomly.

2. **Repeat** until convergence (centroids do not change significantly):
   
   2.1. **Assignment Step:**  
   For each data point $x_i$, assign it to the nearest centroid:
   $$
   c_i \gets \arg\min_{k} \| x_i - \mu_k \|^2
   $$
   
   2.2. **Update Step:**  
   Recompute each centroid as the mean of all points assigned to it:
   $$
   \mu_k \gets \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i
   $$
   where $C_k = \{ x_i : c_i = k \}$

3. **Return:** cluster assignments $C$ and centroids $\{\mu_1, \dots, \mu_K\}$.

### Recall: 
$\arg\min_{k} \| x_i - \mu_k \|^2$ returns the value of k at which the function $\| x_i - \mu_k \|^2$ obtains a minimum. Where here the function is the generalized Euclidean distance or 2-norm between the vector $x$ and the centroids $u$. And in this case n is the number of points in a given cluster k.

$$
\mathbf{x}=(x_1,x_2,\dots,x_n),
\qquad
\mathbf{u}=(u_1,u_2,\dots,u_k).
$$

$$
d(\mathbf{x},\mathbf{u})=
\sqrt{(x_1-u_k)^2+(x_2-u_k)^2+\cdots+(x_n-u_k)^2}
$$ as $k$ runs from 1 to K(total number of clusters).

# Linear Regression

# General $n$-Dimensional Least Squares Regression

We are given $n$ data points  
$x_i \in \mathbb{R}^d$, $y_i \in \mathbb{R}$, for $i=1,\dots,n$.

Define the design matrix

$$
X =
\begin{bmatrix}
x_1^T \\
x_2^T \\
\vdots \\
x_n^T
\end{bmatrix}
\in \mathbb{R}^{n\times d}.
$$

and the response vector

$$
y =
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_n
\end{bmatrix}
\in \mathbb{R}^n.
$$

We seek a parameter vector

$$
w \in \mathbb{R}^d
$$

such that the linear model

$$
\hat y = Xw
$$

best approximates $y$.

---

## Least Squares Objective

Define the residual vector

$$
r(w) = Xw - y.
$$

The least squares problem is

$$
\min_{w\in\mathbb{R}^d} \|Xw - y\|^2.
$$

Expanding,

$$
\|Xw - y\|^2
= (Xw-y)^T(Xw-y)
= w^T X^T X w - 2 y^T X w + y^T y.
$$

---

## Normal Equations

Take the gradient with respect to $w$:

$$
\nabla_w = 2X^TXw - 2X^Ty.
$$

Setting the gradient equal to zero gives

$$
X^TXw = X^Ty.
$$

These are called the **normal equations**.

---

## Closed-Form Solution

If $X^TX$ is invertible, the unique solution is

$$
w = (X^TX)^{-1}X^Ty.
$$

If $X^TX$ is singular, the solution is given by the **Moore–Penrose pseudoinverse**

$$
w = X^+ y.
$$

---

## Geometric Interpretation

The fitted vector $Xw$ is the orthogonal projection of $y$ onto the column space of $X$.  
The residual satisfies

$$
X^T(Xw-y)=0,
$$

so the error is orthogonal to every column of $X$.

---

## Numerical Algorithms

In practice one avoids forming $(X^TX)^{-1}$ directly.

### QR Decomposition

$$
X = QR,
$$

where $Q\in\mathbb{R}^{n\times d}$ has orthonormal columns and  
$R\in\mathbb{R}^{d\times d}$ is upper triangular.  
Solve

$$
Rw = Q^T y.
$$

### Singular Value Decomposition

$$
X = U\Sigma V^T.
$$

Then

$$
w = V\Sigma^{-1}U^T y.
$$


# MORE

# Extensions of Least Squares

## Gradient Descent Formulation

Instead of solving the normal equations, we can minimize

$$
f(w) = \|Xw - y\|^2
$$

iteratively.

The gradient is

$$
\nabla f(w) = 2X^T(Xw - y).
$$

The gradient-descent update rule is

$$
w_{k+1} = w_k - \eta \nabla f(w_k)
$$

which gives

$$
w_{k+1}
= w_k - 2\eta X^T(Xw_k - y).
$$

Here $\eta > 0$ is the learning rate.

This converges to the least-squares solution if $\eta$ is chosen small enough.

---

## Ridge Regression (Regularized Least Squares)

To prevent overfitting or deal with singular $X^TX$, add an $\ell_2$ penalty:

$$
\min_w \; \|Xw - y\|^2 + \lambda \|w\|^2
$$

where $\lambda \ge 0$ is the regularization parameter.

Expanding,

$$
f(w) = (Xw-y)^T(Xw-y) + \lambda w^T w.
$$

Taking the gradient:

$$
\nabla f(w)
= 2X^TXw - 2X^Ty + 2\lambda w.
$$

Setting this equal to zero gives

$$
(X^TX + \lambda I)w = X^T y.
$$

So the ridge-regression solution is

$$
w = (X^TX + \lambda I)^{-1} X^T y.
$$

---

## Geometric Meaning of Ridge Regression

Ordinary least squares projects $y$ onto the column space of $X$.

Ridge regression shrinks the solution toward the origin by penalizing large $\|w\|$, stabilizing the inverse and reducing variance.

---
# PCA

# Principal Component Analysis (PCA)

Let  
$$
x_1,\dots,x_n\in\mathbb{R}^d
$$  
be data points.

---

## Centering

$$
\mu=\frac1n\sum_{i=1}^n x_i
$$

$$
\tilde x_i=x_i-\mu
$$

$$
X=
\begin{bmatrix}
\tilde x_1^T\\
\tilde x_2^T\\
\vdots\\
\tilde x_n^T
\end{bmatrix}
\in\mathbb{R}^{n\times d}
$$

---

## Covariance matrix

$$
\Sigma=\frac1nX^TX
$$

---

## Optimization problem

$$
\max_w w^T\Sigma w
\quad\text{subject to}\quad
w^Tw=1
$$

---

## Lagrangian

$$
\mathcal L(w,\lambda)=w^T\Sigma w-\lambda(w^Tw-1)
$$

$$
\nabla_w\mathcal L=2\Sigma w-2\lambda w=0
$$

$$
\Sigma w=\lambda w
$$

---

## Principal components

The principal components are the eigenvectors of $\Sigma$.
The first component corresponds to the largest eigenvalue.

---

## Dimensionality reduction

Let

$$
W_k=[w_1,\dots,w_k]
$$

Then the reduced data is

$$
Z=XW_k
$$

# more


# Principal Component Analysis (PCA) with Explicit Covariance Matrix

Let $x_1, \dots, x_n \in \mathbb{R}^d$ be data points.

---

## 1. Center the data

Compute the mean vector:

$$
\mu = \frac{1}{n} \sum_{i=1}^n x_i
$$

Define centered vectors:

$$
\tilde x_i = x_i - \mu
$$

Stack them as rows of a data matrix:

$$
X = 
\begin{bmatrix}
\tilde x_1^T \\
\tilde x_2^T \\
\vdots \\
\tilde x_n^T
\end{bmatrix}
\in \mathbb{R}^{n \times d}
$$

---

## 2. Covariance matrix (explicit form)

The sample covariance matrix is

$$
\Sigma = \frac{1}{n} \sum_{i=1}^n \tilde x_i \tilde x_i^T
$$

In **matrix form**, using $X$:

$$
\Sigma = \frac{1}{n} X^T X
$$

If we write it **entry-wise**, for $1 \le j,k \le d$:

$$
\Sigma_{jk} = \frac{1}{n} \sum_{i=1}^n (\tilde x_i)_j (\tilde x_i)_k
$$

where $(\tilde x_i)_j$ is the $j$-th coordinate of the centered vector $\tilde x_i$.

So explicitly, the covariance matrix is:

$$
\Sigma =
\frac{1}{n} 
\begin{bmatrix}
\sum_{i=1}^n (\tilde x_i)_1 (\tilde x_i)_1 & \sum_{i=1}^n (\tilde x_i)_1 (\tilde x_i)_2 & \cdots & \sum_{i=1}^n (\tilde x_i)_1 (\tilde x_i)_d \\
\sum_{i=1}^n (\tilde x_i)_2 (\tilde x_i)_1 & \sum_{i=1}^n (\tilde x_i)_2 (\tilde x_i)_2 & \cdots & \sum_{i=1}^n (\tilde x_i)_2 (\tilde x_i)_d \\
\vdots & \vdots & \ddots & \vdots \\
\sum_{i=1}^n (\tilde x_i)_d (\tilde x_i)_1 & \sum_{i=1}^n (\tilde x_i)_d (\tilde x_i)_2 & \cdots & \sum_{i=1}^n (\tilde x_i)_d (\tilde x_i)_d
\end{bmatrix}
$$

---

## 3. Variance maximization problem

PCA finds a unit vector $w \in \mathbb{R}^d$ that maximizes the variance of the projections:

$$
\max_{w} w^T \Sigma w
\quad \text{subject to} \quad w^T w = 1
$$

The Lagrangian is

$$
\mathcal L(w,\lambda) = w^T \Sigma w - \lambda (w^T w - 1)
$$

Taking the gradient and setting it to zero:

$$
\Sigma w = \lambda w
$$

The **principal components** are the eigenvectors of $\Sigma$, ordered by decreasing eigenvalue.

---

# KNN

### K-Nearest Neighbors (KNN)

The **k-Nearest Neighbors (KNN) algorithm** is a non-parametric, instance-based supervised learning method for **classification** and **regression**. It does not construct a model during training; all computations are performed at prediction time.

Let the training dataset be  

$$
X = \{x_1, x_2, \dots, x_n\} \subset \mathbb{R}^d,
$$

with corresponding labels  

$$
Y = \{y_1, y_2, \dots, y_n\}.
$$

For a query point  

$$
x_q \in \mathbb{R}^d,
$$  

the KNN algorithm proceeds as follows:

---

#### 1. Compute Distances

Compute the distance between the query point and all training points. Using Euclidean distance:

$$
d(x_q, x_i) = \sqrt{\sum_{j=1}^{d} (x_{qj} - x_{ij})^2}, \quad i = 1,2,\dots,n
$$

Other distance metrics (Manhattan, Minkowski) can also be used depending on the application.

---

#### 2. Identify Nearest Neighbors

Select the $k$ training points with the smallest distances to $x_q$. Denote the indices of these neighbors as  

$$
\mathcal{N}_k(x_q) \subset \{1, 2, \dots, n\}.
$$

---

#### 3. Make Predictions

**Classification:** choose the class that occurs most frequently among the neighbors:

$$
\hat{y}_q = \arg\max_{c} \sum_{i \in \mathcal{N}_k(x_q)} \mathbf{1}_{\{y_i = c\}}
$$

where $\mathbf{1}_{\{y_i = c\}}$ is the indicator function, equal to 1 if $y_i = c$ and 0 otherwise.

**Regression:** compute the average of the neighbors’ values:

$$
\hat{y}_q = \frac{1}{k} \sum_{i \in \mathcal{N}_k(x_q)} y_i
$$

---

#### 4. Notes

1. KNN **does not require training**; it is a lazy learner storing all data.  
2. The choice of $k$ affects the **bias-variance tradeoff**:  
   - Small $k$ → low bias, high variance (sensitive to noise).  
   - Large $k$ → high bias, low variance (smoother decision boundary).  
3. **Feature scaling** is important because distances are sensitive to magnitude differences between features.
