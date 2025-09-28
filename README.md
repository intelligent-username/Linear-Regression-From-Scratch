# Linear Regression From Scratch

![Cover](img.jpg)

---

## Outline

- [Linear Regression From Scratch](#linear-regression-from-scratch)
  - [Outline](#outline)
  - [The Math Behind Linear Regression](#the-math-behind-linear-regression)
    - [The Simple Idea](#the-simple-idea)
    - [3D \& Higher Dimensions (The General Case)](#3d--higher-dimensions-the-general-case)
      - [3-Dimensional Formula](#3-dimensional-formula)
      - [$n$-dimensional Form](#n-dimensional-form)
  - [Installation](#installation)
  - [API](#api)
  - [License](#license)

---

## The Math Behind Linear Regression

### The Simple Idea

Linear regression fits a straight line to the data points that best predicts an outcome.

Say you have $n$ points $(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)$ with $n \geq 2$. The goal is to find the line that best "fits" these points. By 'best fit', we mean the line that most accurately predicts a $y$ value from some given $x$ value. This is calculated using a loss function, in this case, I will be using Mean Squared Error (**MSE**). By minimizing error, we maximize accuracy. The sum of squared residuals (errors), our loss function, finds the mean squared error by using the formula:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Where:

- $y_i$ is the actual value from our data.
- $\hat{y}_i$ is the predicted value from our linear model.
- Each term $(y_i - \hat{y}_i)^2$ is the squared difference between the actual and predicted values. We square this difference to ensure that positive and negative differences don't cancel each other out.

We divide by $n$, the number of points, to get the mean of the residuals. This is just the same as calculating the variance of the residuals.

\* Note: When doing statistics, if we're only taking a sample and not the whole population, we would divide by $n-1$ instead. This is to correct for the added bias. For more, read on [Bessel's correction](https://en.wikipedia.org/wiki/Bessel%27s_correction). In Loss Functions, we just divide by $n$.

$$\hat{y}_i = \beta_0 + \beta_1 x$$

is the formula for the predicted $\hat{y}_i$, with $\beta_1$ being the slope and $\beta_0$ being the intercept of the graph.

Finding the optimal values of $\beta_1$ and $\beta_0$ is the goal of linear regression.

Now, in the simple case, we have a manageable finite number of points, through which we can draw a line. Our parameters $\beta_1$ and $\beta_0$ can be easily calculated:

$$
\beta_1 = \frac{\sum_{i=1}^{n}(x_i - \overline{x}) \times (y_i - \overline{y})}{\sum_{i=1}^{n}(x_i - \overline{x})^2}
$$

Here, we divide the covariance of $x$ and $y$ by the variance of $x$ to find $\beta_1$. For more on how this formula is derived, see the case in 3D.

Then, we find $\beta_0$ with the formula:

$$
\beta_0 = \overline{y} - \beta_1\overline{x}
$$

\*Note: Setting the y-intercept this way ensures the line crosses the mean of $y$.

Where:

- $\overline{x}$ is the average value for x
- $\overline{y}$ is the average value for y
- $x_i$ is the observed value of x at that point
- $y_i$ is the observed value of y at that point

This case, where our points are in 2D, is called **Simple Linear Regression**.

There are two more cases to consider in simple linear regression:

- When we have points in higher dimensions (more than one feature)
- When we have a very large number of points (we cannot feasibly calculate the sums)

---

### 3D & Higher Dimensions (The General Case)

</br>

#### 3-Dimensional Formula

In 3D, we are plotting the points $(x,y,z)$. The formula for the line is now:

$$
z = \beta_0 + \beta_1 x + \beta_2 y
$$

This is the same as the 2D formula, just with an extra variable to account for the new dimension.

Now, the loss function is the same, just with $z$ instead of $y$ for clarity:

$$
\begin{align*}
MSE = L(\beta_0, \beta_1, \beta_2) &= \frac{1}{n} \sum_{i=1}^{n} (z_i - \hat{z}_i)^2 \\
&= \frac{1}{n} \sum_{i=1}^{n} (z_i - (\beta_0 + \beta_1 x_i + \beta_2 y_i))^2
\end{align*}
$$

Once again, $\hat{z}_i$ is the predicted value from our linear model, and, for each prediction, we calculated the MSE and try to minimize it.

To find $\beta_0$, $\beta_1$, and $\beta_2$, we take the partial derivatives of the loss function, $L$, with respect to each parameter and set the partial derivatives to zero. We do this to find the global minimum of the loss function.

\*Note: Since the loss function is a quadratic with a non-negative leading coefficient ($n\geq2$), it's guaranteed to have a single local minimum, which is also the global minimum.

Now, we have:

$$
\nabla L = \begin{pmatrix}
\frac{\partial L}{\partial \beta_0} \\
\frac{\partial L}{\partial \beta_1} \\
\frac{\partial L}{\partial \beta_2}
\end{pmatrix}
$$

with

$$
\begin{align*}
\frac{\partial L}{\partial \beta_0} &= \frac{1}{n} \sum_{i=1}^{n} 2 \cdot (z_i - (\beta_0 + \beta_1 x_i + \beta_2 y_i)) \cdot (-1) \\
&= \frac{-2}{n} \sum_{i=1}^{n} (z_i - (\beta_0 + \beta_1 x_i + \beta_2 y_i))
\end{align*}
$$

Similarly,

$$
\begin{align*}
\frac{\partial L}{\partial \beta_1} &= \frac{1}{n} \sum_{i=1}^{n} 2 \cdot (z_i - (\beta_0 + \beta_1 x_i + \beta_2 y_i)) \cdot (-x_i) \\
&= \frac{-2}{n} \sum_{i=1}^{n} (z_i - (\beta_0 + \beta_1 x_i + \beta_2 y_i)) x_i
\end{align*}
$$

$$
\begin{align*}
\frac{\partial L}{\partial \beta_2} &= \frac{1}{n} \sum_{i=1}^{n} 2 \cdot (z_i - (\beta_0 + \beta_1 x_i + \beta_2 y_i)) \cdot (-y_i) \\
&= \frac{-2}{n} \sum_{i=1}^{n} (z_i - (\beta_0 + \beta_1 x_i + \beta_2 y_i)) y_i
\end{align*}
$$

Now, equating to 0 (for the minimum):

$$
\frac{-2}{n} \sum_{i=1}^{n} (z_i - (\beta_0 + \beta_1 x_i + \beta_2 y_i)) = 0 \\
\Rightarrow \sum_{i=1}^{n} (z_i - (\beta_0 + \beta_1 x_i + \beta_2 y_i)) = 0 \\
\Rightarrow \sum_{i=1}^{n} z_i = \sum_{i=1}^{n} (\beta_0 + \beta_1 x_i + \beta_2 y_i) \\
\\
\Rightarrow \sum_{i=1}^{n} z_i = n\beta_0 + \beta_1\sum_{i=1}^{n} x_i +  \beta_2\sum_{i=1}^{n} y_i
$$

Similarly, we have:

$$
\sum_{i=1}^{n} (z_i - (\beta_0 + \beta_1 x_i + \beta_2 y_i)) x_i = 0 \\
\Rightarrow \sum_{i=1}^{n} z_i x_i = \sum_{i=1}^{n} (\beta_0 x_i + \beta_1 x_i^2 + \beta_2 y_i x_i) \\
\Rightarrow \sum_{i=1}^{n} z_i x_i = \beta_0 \sum_{i=1}^{n} x_i + \beta_1 \sum_{i=1}^{n} x_i^2 + \beta_2 \sum_{i=1}^{n} x_i y_i
$$

and

$$
\sum_{i=1}^{n} z_i y_i = \beta_0\sum_{i=1}^{n} (y_i) + \beta_1 \sum_{i=1}^{n} (x_i y_i) + \beta_2 \sum_{i=1}^{n} (y_i^2)
$$

By solving this system of equations, we can find the optimal values for $\beta_0$, $\beta_1$, and $\beta_2$ that minimize the loss function.

For conciseness, this system is often represented in matrix form, with normal equations:

$$
\mathbf{X}\vec{\beta} = \vec{z}
$$

with

$$
\mathbf{X} = \sum_{i=1}^{n} \begin{pmatrix}
n & x_i & y_i \\
x_i & x_i^2 & x_i y_i \\
y_i & x_i y_i & y_i^2
\end{pmatrix}, \
\vec{\beta} = \begin{pmatrix}
\beta_0 \\
\beta_1 \\
\beta_2
\end{pmatrix}, \
\vec{z} = \sum_{i=1}^{n} \begin{pmatrix}
z_i \\
z_i x_i \\
z_i y_i
\end{pmatrix}.
$$

Provided $\mathbf{X}$ is invertible (that is, the predictors $x$ and $y$ are not collinear), the unique solution is:

$$
\vec{\beta} = \mathbf{X}^{-1} \vec{z}.
$$

The vector $\vec{\beta}$ gives the intercept $\beta_0$ and slopes $\beta_1,\beta_2$ of the best-fitting plane:

$$
\hat{z} = \beta_0 + \beta_1 x + \beta_2 y.
$$

\*Note: In more conventional notation, the system of equations derived above is written compactly using matrix transposes as:

\[
X^\top X\,\beta = X^\top z,
\]

where \(X\) is the design matrix containing the predictors and a column of ones for the intercept, and \(z\) is the vector of observed outputs. This is called the normal equation for ordinary least squares. Solving it yields the closed‑form solution:

\[
\beta = (X^\top X)^{-1}X^\top z.
\]

This matrix form is equivalent to the expanded system above; the transposes arise naturally when taking derivatives of the loss function, and this form generalizes directly to any number of predictors.

This procedure generalizes to give the:

#### $n$-dimensional Form

$$
\vec{\beta} = (X^\top X)^{-1} X^\top \vec{y},
$$

Now, the points have $m$ features, so each point is of the form $(x_1, x_2, \ldots, x_m, y)$, with $y$ being the feature we want to predict.

The math and procedure behind this formula is exactly the same as three (and two) dimensions, just more generalized.

---

To review, when performing linear regression, we:

1) Define a loss function.
2) Find the gradient of the loss function.
3) Set this gradient to zero and rearrange to form a system of equations.
4) Write this system in matrix form.
5) Substitute the points to solve for the parameters.

Now that we understand the math behind this, we can start to always skip to step 5 (immediately plug in the points), assuming we always use the same Loss Function.

This method fails when:

- There is no correlation between the variables (obviously).
- There are more parameters than points (for example when there are 5 features but only 3 points).
- The features are linearly dependent (collinear).

(This is because the matrix $\mathbf{X}$ becomes non-invertible, and thus we cannot solve for $\vec{\beta}$ uniquely.)

Conversely, if we have:

- More points than parameters
- No collinearity
- Some correlation

Linear regression can always produce a line of best fit in theory.

However, in practice, there are still some issues.

- The relationship isn't linear. This calls for **polynomial regression** to be used instead.
- Overfitting: the model may capture noise instead of the underlying trend. This can be mitigated with regularization and other **complexity control techniques**.
- Unrealistic assumptions: linear regression assumes homoscedasticity (constant variance of errors) and normality of errors, which may not hold in real-world data. This can be solved by using **Generalized Least Squares** methods.
- Infeasible number of points: when there are too many points, calculating the sums becomes too costly. This calls for more estimative methods like **Gradient Descent** and its variations.

---

## Installation

Clone this repo:

```powershell
git clone https://github.com/intelligent-username/Linear-Regression-From-Scratch.git
```

Install dependencies.

With Pip:

```powershell
cd Linear-Regression-From-Scratch
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
pip install -e .  
```

With Conda:

```powershell
conda env create -f environment.yml; 
conda activate naivelinear
```

(and

```powershell
pip install -e
```

if you want the tests to work)

Install via editable local clone (package name: `naivelinear`):

Run tests (optional):

```powershell
pytest -q
```

Basic usage (functional API):

```python
from naivelinear import linear_regression

points = [
    (1.0, 2.0, 10.5),  # (x1, x2, y)
    (2.0, 0.5, 11.2),
    (3.0, 1.5, 14.0),
]

params = linear_regression(points)
print("Parameters (bias first):", params)
```

Class API:

```python
from naivelinear import LinearRegressionNaive

model = LinearRegressionNaive().fit(points)
print(model.params_)
pred = model.predict([[2.5, 1.0]])
```

## API

| Object | Description |
|--------|-------------|
| `linear_regression(points)` | Returns parameter vector (bias first) using normal equation. |
| `LinearRegressionNaive` | Estimator with `fit`, `predict`, and simple `mean_squared_error`. |
| `mean_squared_error(y_true, y_pred)` | Basic MSE metric. |

## License

Distributed under the [MIT License](LICENSE).

---
