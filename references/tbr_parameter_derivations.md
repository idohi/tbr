## 📚 Table of Contents

- [🔗 Relevant Links and Sources](#-relevant-links-and-sources)
- [📘 Mapping Between Python Variables, R Variables, and Mathematical Notation](#-mapping-between-python-variables-r-variables-and-mathematical-notation)
- [🔢 Notation](#-notation)
- [📘 Standard Deviation of the Sum of Counterfactual Predictions (In R package `y.pred.sd`)](#-standard-deviation-of-the-sum-of-counterfactual-predictions-in-r-package-ypredsd)
  - [🧭 How to interpret this connection:](#-how-to-interpret-this-connection)
- [📘 Derive the variance of $y_*$](#-derive-the-variance-of-y_-mathbbvy_)
- [📘Derive the variance of $\hat{y}_*$](#-derive-the-variance-of-haty_-mathbbvhaty_)
  - [Derive the variance of $\hat{\beta}_1$](#derive-the-variance-of-hatbeta_1-mathbbvhatbeta_1)
  - [Derive the variance of $\hat{\beta}_0$](#derive-the-variance-of-hatbeta_0-mathbbvhatbeta_0)
  - [Compute Each Term](#compute-each-term)
  - [Derive the covariance between $\hat{\beta}_0$ and $\hat{\beta}_1$ ](#derive-the-covariance-between-hatbeta_0-and-hatbeta_1-textcovhatbeta_0-hatbeta_1)
  - [Combine Results](#combine-results)
- [📘 Variance of the Sum of Counterfactual Predictions (In R package `y.pred.var.cum.test`)](#-variance-of-the-sum-of-counterfactual-predictions-in-r-package-ypredvarcumtest)
- [📘 Posterior Variance of the Estimated Causal Effect (In R package `y.pred.var.cum`)](#-posterior-variance-of-the-estimated-causal-effect-in-r-package-ypredvarcum)
  - [🔍 Note on the Assumption in the R Package](#-note-on-the-assumption-in-the-r-package)
- [📘 Posterior Standard Deviation of the Estimated Causal Effect (In R package `y.pred.sd.cum`)](#-posterior-standard-deviation-of-the-estimated-causal-effect-in-r-package-ypredsdcum)
- [📘 Standard Deviation of the Fitted Value $\hat{y}_t$ (R package: `y.hat.sd`)](#-standard-deviation-of-the-fitted-value-haty_t-r-package-yhatsd)
- [📘 Distinguishing Between Variance of $\hat{y}$ and Variance of $y$](#-distinguishing-between-variance-of-haty-and-variance-of-y-)
- [📘 Credible Interval for the Posterior Distribution of $\Delta r(T)$](#-credible-interval-for-the-posterior-distribution-of-delta-rt)
- [📘 Interpretation of TBR Summary Output](#-interpretation-of-tbr-summary-output)
  - [🔹 `estimate`](#-estimate)
  - [🔹 `precision`](#-precision)
  - [🔹 `lower` and `upper`](#-lower-and-upper)
  - [🔹 `se`](#-se)
  - [🔹 `level`](#-level)
  - [🔹 `thres`](#-thres)
  - [🔹 `prob`](#-prob)
  - [🔹 `model`](#-model)
  - [🔹 `alpha` and `beta`](#-alpha-and-beta)
  - [🔹 `alpha_beta_cov`](#-alpha_beta_cov)
  - [🔹 `sigma`](#-sigma)
  - [🔹 `t_dist_df`](#-t_dist_df)
- [📘 Posterior Variance and Credible Interval for a Subinterval of the Test Period](#-posterior-variance-and-credible-interval-for-a-subinterval-of-the-test-period)
- [📎 Appendix](#-appendix)

## 🔗 Relevant Links and Sources

- 📄 [TBR Paper – *Estimating Ad Effectiveness using Geo Experiments in a Time-Based Regression Framework*](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45950.pdf)
- 📦 [GeoexperimentsResearch R Package (rdrr.io view)](https://rdrr.io/github/google/GeoexperimentsResearch/)
- 💻 [R Implementation of `summary.TBRAnalysisFitTbr1()` on GitHub](https://github.com/google/GeoexperimentsResearch/blob/master/R/summary_tbranalysisfit_tbr1.R)
- 💻 [R Implementation of `DoTBRAnalysis()` on GitHub](https://github.com/google/GeoexperimentsResearch/blob/master/R/dotbranalysis.R)
- 💻 [R Implementation of `DoTBRAnalysis_TBR1()` on GitHub](https://github.com/google/GeoexperimentsResearch/blob/master/R/dotbranalysis_tbr1.R)
- 📁 Local R file in this project:
  - `tbr_function.R`  
    - Calls: [dotbranalysis.R](https://github.com/google/GeoexperimentsResearch/blob/master/R/dotbranalysis.R)  
      - Calls: [dotbranalysis_tbr1.R](https://github.com/google/GeoexperimentsResearch/blob/master/R/dotbranalysis_tbr1.R)

- 🐍 Python–R interface:
  - `run_and_collect_tbr_output()` function in `tbr_run.py`


## 📘 Mapping Between Python Variables, R Variables, and Mathematical Notation

The table below provides a reference for how variables from the Python code correspond to the internal variables used in the [GeoexperimentsResearch R package](https://rdrr.io/github/google/GeoexperimentsResearch/), and how they relate to the mathematical notation used in this document.


| Python Variable | R Variable        | Mathematical Notation                            |
|-----------------|-------------------|--------------------------------------------------|
| `date`          | `kDate`           | — (timestamp, no direct math symbol)             |
| `period`        | `kPeriod`         | — (indicator for pre-test, test, etc.)           |
| `y`             | `kY`              | $y_t$ (and $y_*$ in TBR test period)             |
| `x`             | `kX`              | $x_t$                                            |
| `pred`          | `y.hat`           | $\hat{y}_t$ (and $\hat{y}_*$ in TBR test period) |
| `predsd`        | `y.pred.sd`       | $\sqrt{\mathbb{V}[y_t]}$                         |
| `dif`           | `y.pred.dif`      | $y_t - \hat{y}_t$                                |
| `cumdif`        | `y.cum.dif`       | $\sum (y_t - \hat{y}_t)$                         |
| `cumsd`         | `y.pred.sd.cum`   | $\sqrt{\mathbb{V}[\sum (y_t - \hat{y}_t)]}$      |
| `estsd`         | `y.hat.sd`        | $\sqrt{\mathbb{V}[\hat{y}_t]}$                   |

**Note:** In the TBR test period, replace $\hat{y}_t$ and $y_t$ with $\hat{y}_*$ and $y_*$ respectively in all expressions.



## 🔢 Notation

We use the symbol $\mathbb{V}[\cdot]$ to denote both the theoretical variance (e.g., $\sigma^2$) and its sample-based estimator (e.g., $s^2$) when context permits.

This is done for clarity and consistency of notation throughout the derivation.

## 📘 Standard Deviation of the Sum of Counterfactual Predictions (In R package `y.pred.sd`)


This section provides a complete mathematical derivation of the formula used to calculate the **standard error of prediction** in simple linear regression.

It explains, step by step, how to derive the total prediction variance for a new input point $x_*$, leading to the final formula:

$$
\boxed{
\sqrt{ \mathbb{V}[y_*] } = s \cdot \sqrt{ 1 + \frac{1}{n} + \frac{(x_* - \bar{x})^2}{\sum_{i=1}^n (x_i - \bar{x})^2} }
}
$$

This formula corresponds directly to the following line in the R implementation of `DoTBRAnalysis()` in the [GeoexperimentsResearch package](https://rdrr.io/github/google/GeoexperimentsResearch/src/R/dotbranalysis_tbr1.R):

```
y.pred.sd <- sqrt(y.hat.sd^2 + sigma^2)  # Pointwise s.d. of the predictions
```


For more details on the derivation of the formula, see the section [Derive the variance of $y_*$ ($\mathbb{V}[y_*]$)](#derive-the-variance-of-y_-mathbbvy_) below.

---

## 📘 Derive the variance of $y_*$ ($\mathbb{V}[y_*]$)

Linear regression model:

$$y_i = \beta_0 + \beta_1 x_i + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma^2)$$


OLS (Ordinary Least Squares) estimators:

$$\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i, \quad \bar{y} = \frac{1}{n} \sum_{i=1}^n y_i$$

$$S_{xx} = \sum_{i=1}^n (x_i - \bar{x})^2$$

Taking the derivative of the error function with respect to $\beta_0$ and $\beta_1$ and setting it to zero gives us the OLS estimators:

And because  $\sum_{i=1}^n (x_i - \bar{x})=0$ , we get:

$$\hat{\beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{S_{xx}}=\frac{\sum_{i=1}^n (x_i - \bar{x})y_i}{S_{xx}}$$ 

$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

Fitted Values and Residuals

$$\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$$

$$\varepsilon_i = y_i - \hat{y}_i$$

$$\mathbb{V}[\varepsilon_i] = \sigma^2$$

The variance of the residuals which is an unbiased estimator of the variance of the errors, $\sigma^2$, is given by:

$$s^2 = \frac{1}{n - 2} \sum_{i=1}^n \varepsilon_i^2 = \frac{1}{n - 2} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

The predicted value at a new input point $x_*$ is given by:

$$\hat{y}_* = \hat{\beta}_0 + \hat{\beta}_1 x_*$$

---

Variance of $\hat{y}_*$

$$
\mathbb{V}[\hat{y}_*] = s^2 \left( \frac{1}{n} + \frac{(x_* - \bar{x})^2}{\sum_{i=1}^n (x_i - \bar{x})^2} \right)
$$ 
For more details on the derivation of this formula, see the section [Derive the variance of $\hat{y}_*$](#derive-the-variance-of-haty_-mathbbvhaty_) below.

<br></br>
Assume:

$$
y_* = \hat{y}_* + \varepsilon_*, \quad \varepsilon_* \sim \mathcal{N}(0, \sigma^2)
$$

$\hat{y}_*$ and $\varepsilon_*$ are independet, then:

$$
\mathbb{V}[y_*] = \mathbb{V}[\hat{y}_* +\varepsilon_*] = \mathbb{V}[\hat{y}_*] + \sigma^2
$$

Replace $\sigma^2$ with $s^2$:

$$
\mathbb{V}[y_*] = s^2 \left( 1 + \frac{1}{n} + \frac{(x_* - \bar{x})^2}{\sum_{i=1}^n (x_i - \bar{x})^2} \right)
$$


Final formula:

$$
\boxed
{
\sqrt{ \mathbb{V}[y_*] } = \sqrt{\mathbb{V}[\hat{y}_*] + s^2} = s \cdot \sqrt{ 1 + \frac{1}{n} + \frac{(x_* - \bar{x})^2}{\sum_{i=1}^n (x_i - \bar{x})^2} }
}
$$

---

## 📘 Derive the variance of $\hat{y}_*$ ($\mathbb{V}[\hat{y}_*]$)

Predicted Value at $x_*$:

$\hat{y}_* = \hat{\beta}_0 + \hat{\beta}_1 x_*$

$\mathbb{V}[\hat{y}_*] = \mathbb{V}[\hat{\beta}_0 + \hat{\beta}_1 x_*]$

Use variance of linear combinations:

$\mathbb{V}[aX + bY] = a^2 \mathbb{V}[X] + b^2 \mathbb{V}[Y] + 2ab \, \text{Cov}(X,Y)$
<br></br>

$\mathbb{V}[\hat{\beta}_0 + \hat{\beta}_1 x_*] = \mathbb{V}[\hat{\beta}_0] + x_*^2 \mathbb{V}[\hat{\beta}_1] + 2x_* \, \text{Cov}(\hat{\beta}_0, \hat{\beta}_1)$

Thus:

$\mathbb{V}[\hat{y}_*] = \mathbb{V}[\hat{\beta}_0] + x_*^2 \mathbb{V}[\hat{\beta}_1] + 2x_* \, \text{Cov}(\hat{\beta}_0, \hat{\beta}_1)$

For more details on the derivations of $\mathbb{V}[\hat{\beta}_0]$, $\mathbb{V}[\hat{\beta}_1]$, and $\text{Cov}(\hat{\beta}_0, \hat{\beta}_1)$, see the following links:
- [Derive the variance of $\hat{\beta}_1$ ($\mathbb{V}[\hat{\beta}_1]$)](#derive-the-variance-of-hatbeta_1-mathbbvhatbeta_1)
- [Derive the variance of $\hat{\beta}_0$ ($\mathbb{V}[\hat{\beta}_0]$)](#derive-the-variance-of-hatbeta_0-mathbbvhatbeta_0)
- [Derive the covariance between $\hat{\beta}_0$ and $\hat{\beta}_1$ ($\text{Cov}(\hat{\beta}_0, \hat{\beta}_1)$)](#derive-the-covariance-between-hatbeta_0-and-hatbeta_1-textcovhatbeta_0-hatbeta_1)

---

### Derive the variance of $\hat{\beta}_1$ ($\mathbb{V}[\hat{\beta}_1]$)

From the linear regression model:
$$
y_i = \beta_0 + \beta_1 x_i + \varepsilon_i
$$

$$
\mathbb{E}[\varepsilon_i] = 0, \quad \mathbb{V}[\varepsilon_i] = \sigma^2, \quad \text{Cov}(\varepsilon_i, \varepsilon_j) = 0 \text{ for } i \neq j
$$

$$
S_{xx} = \sum_{i=1}^n (x_i - \bar{x})^2
$$

$$
\hat{\beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{S_{xx}}
$$

<br></br>

$$
\frac{1}{n}\sum_{i=1}^n y_i = \frac{1}{n}\sum_{i=1}^n \beta_0 + \beta_1 x_i + \varepsilon_i
$$
Thus:
$$
\bar{y} = \beta_0 + \beta_1 \bar{x} + \bar{\varepsilon}
$$

Now express $y_i$ in terms of the linear model:

$$
y_i - \bar{y} = \beta_1(x_i - \bar{x}) + \varepsilon_i - \bar{\varepsilon}
$$

Substitute into $\hat{\beta}_1$:

$$
\hat{\beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})\left[ \beta_1(x_i - \bar{x}) + \varepsilon_i - \bar{\varepsilon} \right]}{S_{xx}}
$$

Distribute the sum:

$$
\hat{\beta}_1 = \beta_1 + \frac{\sum_{i=1}^n (x_i - \bar{x})(\varepsilon_i - \bar{\varepsilon})}{S_{xx}}
$$

Let us define:

$$
Z := \sum_{i=1}^n (x_i - \bar{x})(\varepsilon_i - \bar{\varepsilon})
$$

Then:

$$
\hat{\beta}_1 = \beta_1 + \frac{Z}{S_{xx}}
$$

So:

$$
\mathbb{V}[\hat{\beta}_1] = \mathbb{V}\left( \frac{Z}{S_{xx}} \right) = \frac{1}{S_{xx}^2} \mathbb{V}(Z)
$$

Expand $Z$:

$$
Z = \sum_{i=1}^n (x_i - \bar{x})(\varepsilon_i - \bar{\varepsilon})
= \sum_{i=1}^n (x_i - \bar{x}) \varepsilon_i - \bar{\varepsilon} \sum_{i=1}^n (x_i - \bar{x})
$$

But:

$$
\sum_{i=1}^n (x_i - \bar{x}) = 0 \Rightarrow Z = \sum_{i=1}^n (x_i - \bar{x}) \varepsilon_i
$$

Thus:

$$
\mathbb{V}(Z) = \mathbb{V}\left( \sum_{i=1}^n (x_i - \bar{x}) \varepsilon_i \right)
$$

Since the $\varepsilon_i$ are uncorrelated we can insert the variance into the sum, and since they have equal variance $\sigma^2$ we get:

$$
\mathbb{V}(Z) = \sum_{i=1}^n (x_i - \bar{x})^2 \mathbb{V}(\varepsilon_i)
= \sum_{i=1}^n (x_i - \bar{x})^2 \cdot \sigma^2 = \sigma^2 S_{xx}
$$

Finally, we have:

$$
\mathbb{V}[\hat{\beta}_1] = \frac{1}{S_{xx}^2} \cdot \sigma^2 S_{xx}
= \frac{\sigma^2}{S_{xx}}
$$

Substituting $S_{xx}$ back in gives us:

$$
\mathbb{V}[\hat{\beta}_1] = \frac{\sigma^2}{\sum_{i=1}^n (x_i - \bar{x})^2}
$$

---

### Derive the variance of $\hat{\beta}_0$ ($\mathbb{V}[\hat{\beta}_0]$)

Recall the OLS estimators:

$$
\hat{\beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2}
$$

$$
\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}
$$

We seek $\mathbb{V}[\hat{\beta}_0]$. Start from this identity:

$$
\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}
$$

We apply:

$$
\mathbb{V}[\hat{\beta}_0] = \mathbb{V}[\bar{y} - \hat{\beta}_1 \bar{x}]
$$

Use variance of linear combinations:

$$
\mathbb{V}[aX + bY] = a^2 \mathbb{V}[X] + b^2 \mathbb{V}[Y] + 2ab \, \text{Cov}(X,Y)
$$

Here:

- $X = \bar{y}, a = 1$
- $Y = \hat{\beta}_1, b = -\bar{x}$

So:

$$
\mathbb{V}[\hat{\beta}_0] = \mathbb{V}[\bar{y}] + \bar{x}^2 \mathbb{V}[\hat{\beta}_1] - 2\bar{x} \, \text{Cov}(\bar{y}, \hat{\beta}_1)
$$

###  Compute Each Term

First term, $\mathbb{V}[\bar{y}]$:
$$
\bar{y} = \frac{1}{n} \sum_{i=1}^n y_i = \frac{1}{n} \sum_{i=1}^n (\beta_0 + \beta_1 x_i + \varepsilon_i)
\Rightarrow
\mathbb{V}[\bar{y}] = \mathbb{V} \left( \frac{1}{n} \sum_{i=1}^n \varepsilon_i \right) = \frac{\sigma^2}{n}
$$

Second term, $\bar{x}^2 \mathbb{V}[\hat{\beta}_1]$:

From earlier:

$$
\mathbb{V}[\hat{\beta}_1] = \frac{\sigma^2}{\sum_{i=1}^n (x_i - \bar{x})^2}
$$

So:

$$
\bar{x}^2 \mathbb{V}[\hat{\beta}_1] = \bar{x}^2 \cdot \frac{\sigma^2}{\sum_{i=1}^n (x_i - \bar{x})^2}
$$

Third term, $- 2\bar{x} \, \text{Cov}(\bar{y}, \hat{\beta}_1)$:

Recall:

$$
\text{Cov}(\bar{y}, \hat{\beta}_1)
= \text{Cov} \left( \frac{1}{n} \sum_{i=1}^n y_i, \frac{\sum_{j=1}^n (x_j - \bar{x}) y_j}{\sum_{j=1}^n (x_j - \bar{x})^2} \right)
$$

Let:

$$
a_i = \frac{1}{n}, \quad b_i = \frac{(x_i - \bar{x})}{\sum_{j=1}^n (x_j - \bar{x})^2}
$$

Then:

$$
\text{Cov}(\bar{y}, \hat{\beta}_1) = \sum_{i=1}^n a_i b_i \mathbb{V}[y_i] = \sum_{i=1}^n \frac{1}{n} \cdot \frac{(x_i - \bar{x})}{\sum_{j=1}^n (x_j - \bar{x})^2} \cdot \sigma^2
$$

$$
= \sigma^2 \cdot \frac{1}{n} \cdot \frac{\sum_{i=1}^n (x_i - \bar{x})}{\sum_{j=1}^n (x_j - \bar{x})^2} = 0
\quad \text{since } \sum_{i=1}^n (x_i - \bar{x}) = 0
$$

So:

$$
\text{Cov}(\bar{y}, \hat{\beta}_1) = 0
$$


Finally, combine all terms:

$$
\mathbb{V}[\hat{\beta}_0] 
= \frac{\sigma^2}{n} + \bar{x}^2 \cdot \frac{\sigma^2}{\sum_{i=1}^n (x_i - \bar{x})^2}
= \sigma^2 \left( \frac{1}{n} + \frac{\bar{x}^2}{\sum_{i=1}^n (x_i - \bar{x})^2} \right)
$$

---

### Derive the covariance between $\hat{\beta}_0$ and $\hat{\beta}_1$ ($\text{Cov}(\hat{\beta}_0, \hat{\beta}_1)$)


Recall Estimators:

$$
\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}
$$

$$
\hat{\beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x}) y_i}{\sum_{i=1}^n (x_i - \bar{x})^2}
$$

$$
\text{Cov}(\hat{\beta}_0, \hat{\beta}_1) = \text{Cov}(\bar{y} - \hat{\beta}_1 \bar{x}, \hat{\beta}_1)
$$

Apply the identity:

$$
\text{Cov}(A + B, C) = \text{Cov}(A, C) + \text{Cov}(B, C)
$$

Yielding:

$$
\text{Cov}(\hat{\beta}_0, \hat{\beta}_1) = \text{Cov}(\bar{y}, \hat{\beta}_1) - \bar{x} \cdot \text{Var}(\hat{\beta}_1)
$$

We already know from earlier that:

$$
\text{Cov}(\bar{y}, \hat{\beta}_1) = 0
$$

and:

$$
\text{Var}(\hat{\beta}_1) = \frac{\sigma^2}{\sum_{i=1}^n (x_i - \bar{x})^2}
$$


Finally, we get:

$$
\text{Cov}(\hat{\beta}_0, \hat{\beta}_1) = - \frac{\bar{x} \sigma^2}{\sum_{i=1}^n (x_i - \bar{x})^2}
$$

---

### Combine Results

$$\mathbb{V}[\hat{y}_*] = \mathbb{V}[\hat{\beta}_0] + x_*^2 \mathbb{V}[\hat{\beta}_1] + 2x_* \, \text{Cov}(\hat{\beta}_0, \hat{\beta}_1)$$

$$\mathbb{V}[\hat{\beta}_0] = \sigma^2 \left( \frac{1}{n} + \frac{\bar{x}^2}{S_{xx}} \right), \quad \mathbb{V}[\hat{\beta}_1] = \frac{\sigma^2}{S_{xx}}, \quad \text{Cov}(\hat{\beta}_0, \hat{\beta}_1) = -\frac{\bar{x} \sigma^2}{S_{xx}}$$

$$\Rightarrow \mathbb{V}[\hat{y}_*] = \sigma^2 \left( \frac{1}{n} + \frac{(x_* - \bar{x})^2}{S_{xx}} \right)$$

Replace $\sigma^2$ with $s^2$:

$$\mathbb{V}[\hat{y}_*] = s^2 \left( \frac{1}{n} + \frac{(x_* - \bar{x})^2}{S_{xx}} \right)$$
<br></br>

---

## 📘 Variance of the Sum of Counterfactual Predictions (In R package `y.pred.var.cum.test`)

In the GeoexperimentsResearch R package , the variable `y.pred.var.cum.test` refers to the **posterior variance of the cumulative counterfactual prediction** during the test period, [GeoexperimentsResearch R package](https://rdrr.io/github/google/GeoexperimentsResearch/src/R/dotbranalysis_tbr1.R).

This is the variance of the sum of predicted values:

$$
\sum_{t=1}^{T} \hat{y}_t^* = \sum_{t=1}^{T} (\hat{\beta}_0 + \hat{\beta}_1 x_t)
$$

We can write:

$$
\sum_{t=1}^{T} \hat{y}_t^* = T \cdot \hat{\beta}_0 + \hat{\beta}_1 \cdot \sum_{t=1}^{T} x_t = T \cdot (\hat{\beta}_0 + \bar{x}_T \hat{\beta}_1)
$$

Where:

- $T$ is the number of test period time points
- $\bar{x}_T = \frac{1}{T} \sum_{t=1}^T x_t$

To compute the variance of this sum, we apply:

$$
\mathbb{V}\left(\sum_{t=1}^{T} \hat{y}_t^*\right) = \mathbb{V}\left( T \cdot (\hat{\beta}_0 + \bar{x}_T \hat{\beta}_1) \right)
= T^2 \cdot \mathbb{V}(\hat{\beta}_0 + \bar{x}_T \hat{\beta}_1)
$$

Using the variance of a linear combination of two random variables:

$$
\mathbb{V}(\hat{\beta}_0 + \bar{x}_T \hat{\beta}_1)
= \mathbb{V}(\hat{\beta}_0) + 2 \bar{x}_T \cdot \text{Cov}(\hat{\beta}_0, \hat{\beta}_1) + \bar{x}_T^2 \cdot \mathbb{V}(\hat{\beta}_1)
$$

For more details on the derivations of $\mathbb{V}[\hat{\beta}_0]$, $\mathbb{V}[\hat{\beta}_1]$, and $\text{Cov}(\hat{\beta}_0, \hat{\beta}_1)$, see the following links:
- [Derive the variance of $\hat{\beta}_1$ ($\mathbb{V}[\hat{\beta}_1]$)](#derive-the-variance-of-hatbeta_1-mathbbvhatbeta_1)
- [Derive the variance of $\hat{\beta}_0$ ($\mathbb{V}[\hat{\beta}_0]$)](#derive-the-variance-of-hatbeta_0-mathbbvhatbeta_0)
- [Derive the covariance between $\hat{\beta}_0$ and $\hat{\beta}_1$ ($\text{Cov}(\hat{\beta}_0, \hat{\beta}_1)$)](#derive-the-covariance-between-hatbeta_0-and-hatbeta_1-textcovhatbeta_0-hatbeta_1)

Then the final expression is:

$$
\boxed{
\mathbb{V}\left(\sum_{t=1}^{T} \hat{y}_t^*\right) = T^2 \left( \mathbb{V}(\hat{\beta}_0) + 2 \bar{x}_T \cdot \text{Cov}(\hat{\beta}_0, \hat{\beta}_1) + \bar{x}_T^2 \cdot \mathbb{V}(\hat{\beta}_1) \right)
}
$$

This is the quantity referred to as `y.pred.var.cum.test` in the package,

`y.pred.var.cum.test` = $\mathbb{V}\left(\sum_{t=1}^{T} \hat{y}_t^*\right)$
<br></br>

## 📘 Posterior Variance of the Estimated Causal Effect (In R package `y.pred.var.cum`)

In the GeoexperimentsResearch R package, the variable `y.pred.var.cum` represents the **posterior variance of the total estimated causal effect** during the test period. It combines two components:

1. The **model uncertainty** in predicting the counterfactual $\hat{y}_t^*$
2. The **residual noise** in the observed outcome $y_t$

The formula in R is:

```r
y.pred.var.cum <- y.pred.var.cum.test + T * sigma^2
```

---
### 🔹 Clarification on the Variance of $y_t$

In the TBR framework, each observed outcome $y_t$ in the test period is treated as a realization from a linear model (not deterministic) of the form:

$$
y_t = \beta_0 + \beta_1 x_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2)
$$

In this context, $\beta_0$, $\beta_1$, and $x_t$ are treated as fixed (non-random) quantities. The randomness in $y_t$ arises solely from the residual noise term $\varepsilon_t$. Therefore, the variance of $y_t$ is:

$$
\mathbb{V}[y_t] = \mathbb{V}[\beta_0 + \beta_1 x_t + \varepsilon_t] = \mathbb{V}[\beta_0 + \beta_1 x_t] + \mathbb{V}[\varepsilon_t] =\sigma^2
$$

We assume that the variance of the noise in $y_t$ is the same as in the pre-test period (assumption of homoscedasticity). The value of $\sigma^2$ is unknown and is estimated from the residual variance of the regression fitted on the pre-test period:

$$
\hat{\sigma}^2 = \frac{1}{n - 2} \sum_{t \in \text{pre}} (y_t - \hat{\beta}_0 - \hat{\beta}_1 x_t)^2
$$

---


In math terms:

$$
\mathbb{V}[\Delta r(T)] = \mathbb{V}\left(\sum_{t=1}^{T} (y_t - \hat{y}_t^*)\right)
= \mathbb{V}\left(\sum_{t=1}^{T} y_t\right) + \mathbb{V}\left(\sum_{t=1}^{T} \hat{y}_t^*\right)
$$


Based on the assumption of i.i.d. residuals $\varepsilon_t$, the $y_t$ values are treated as uncorrelated, so we use $\mathbb{V}\left[\sum y_t\right] = \sum \mathbb{V}[y_t]$ for ease of computation, thus:


- $\mathbb{V}\left(\sum y_t\right) \approx T \cdot \sigma^2$
- $\mathbb{V}\left(\sum \hat{y}_t^*\right) = T^2 \cdot v$ ,&nbsp; where:&nbsp;&nbsp;&nbsp;$v = \mathbb{V}(\hat{\beta}_0) + 2 \bar{x}_T \cdot \text{Cov}(\hat{\beta}_0, \hat{\beta}_1) + \bar{x}_T^2 \cdot \mathbb{V}(\hat{\beta}_1)$


Putting this together:

$$
\boxed{
\mathbb{V}[\Delta r(T)] = T \cdot \sigma^2 + T^2 \cdot v  
}
$$

This is the expression referred to as `y.pred.var.cum` in the R package. It represents the full posterior variance of the **cumulative estimated effect** (not just the counterfactual prediction).

### 🔍 Note on the Assumption in the R Package

In the GeoexperimentsResearch R package, the residual variance component:

$$
\mathbb{V}\left( \sum y_t \right) = T \cdot \sigma^2
$$

is used as an **equality**, not as an approximation.

While this expression is **only exact** if the $y_t$ values are independent and identically distributed (i.i.d.), the R implementation **assumes it holds directly**, i.e., that there is:

- Constant variance across all $y_t$
- No autocorrelation between time steps

This simplification enables efficient computation of credible intervals. The authors of the **TBR paper**  
(*Estimating Ad Effectiveness using Geo Experiments in a Time-Based Regression Framework*) justify its use empirically — in their simulations, the credible intervals still had accurate coverage, **even under correlated or noisy conditions**.

> ✅ So, while mathematically this is an approximation, the R package **treats it as an equality** for practical purposes.

## 📘 Posterior Standard Deviation of the Estimated Causal Effect (In R package `y.pred.sd.cum`)

In the GeoexperimentsResearch R package, the variable `y.pred.sd.cum` (named `cumsd` in the output data frame) represents the **posterior standard deviation of the cumulative causal effect estimate** at each point in the prediction period.

It is computed simply as the square root of the full posterior variance:

$$
y.pred.sd.cum = \sqrt{ y.pred.var.cum }
$$

This value reflects the uncertainty around the cumulative estimated effect $\sum_{t=1}^{T} (y_t - \hat{y}_t^*)$ up to time $T$, and naturally increases over time as more variance components accumulate.

For the detailed derivation of `y.pred.var.cum`, see the section above [Posterior Variance of the Estimated Causal Effect](#-posterior-variance-of-the-estimated-causal-effect-in-r-package-ypredvarcum).


This quantity is used for constructing credible intervals at each point in the test period.

## 📘 Standard Deviation of the Fitted Value $\hat{y}_t$ (R package: `y.hat.sd`)

In linear regression, the fitted value at a given covariate input $x_t$ is given by:

$$
\hat{y}_t = \hat{\beta}_0 + \hat{\beta}_1 x_t
$$

To derive its variance, we apply the formula for the variance of a linear combination:

$$
\mathbb{V}[\hat{y}_t] = \mathbb{V}[\hat{\beta}_0] + x_t^2 \mathbb{V}[\hat{\beta}_1] + 2x_t \cdot \text{Cov}(\hat{\beta}_0, \hat{\beta}_1)
$$

Substituting the known expressions:
- $\mathbb{V}[\hat{\beta}_0] = \sigma^2 \left( \frac{1}{n} + \frac{\bar{x}^2}{S_{xx}} \right)$
- $\mathbb{V}[\hat{\beta}_1] = \frac{\sigma^2}{S_{xx}}$
- $\text{Cov}(\hat{\beta}_0, \hat{\beta}_1) = -\frac{\bar{x} \sigma^2}{S_{xx}}$

For more on the components of this variance, see:
- [Derive the variance of $\hat{\beta}_0$](#derive-the-variance-of-hatbeta_0-mathbbvhatbeta_0)
- [Derive the variance of $\hat{\beta}_1$](#derive-the-variance-of-hatbeta_1-mathbbvhatbeta_1)
- [Derive the covariance between $\hat{\beta}_0$ and $\hat{\beta}_1$](#derive-the-covariance-between-hatbeta_0-and-hatbeta_1-textcovhatbeta_0-hatbeta_1)

We then obtain:

$$
\mathbb{V}[\hat{y}_t] =
\sigma^2 \left( \frac{1}{n} + \frac{\bar{x}^2}{S_{xx}} \right) + x_t^2 \cdot \frac{\sigma^2}{S_{xx}} - 2 x_t \cdot \frac{\bar{x} \sigma^2}{S_{xx}}
= \sigma^2 \left( \frac{1}{n} + \frac{(x_t - \bar{x})^2}{S_{xx}} \right)
$$

Replacing $\sigma^2$ with its unbiased estimator $s^2$, the final formula becomes:

$$
\mathbb{V}[\hat{y}_t] = s^2 \left( \frac{1}{n} + \frac{(x_t - \bar{x})^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2} \right)
$$

Where $S_{xx} = \sum_{i=1}^{n} (x_i - \bar{x})^2$ and $s^2 = \frac{1}{n-2} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$.

The standard deviation of the fitted value (used in the R package as `y.hat.sd`) is:

`y.hat.sd` = $\sqrt{ \mathbb{V}[\hat{y}_t] }$

This expression captures the variability in the model’s estimated response at the point $x_t$, arising only from uncertainty in the estimated regression coefficients.

## 📘 Distinguishing Between Variance of $\hat{y}$ and Variance of $y$ 

When predicting the response at a new point $x_t$, it's important to distinguish between two types of variance:

- **Variance of the model’s estimation at $x_t$**:

  The predicted value is given by:
  $$
  \hat{y}_t = \hat{\beta}_0 + \hat{\beta}_1 x_t
  $$
  Since $\hat{\beta}_0$ and $\hat{\beta}_1$ are estimated from data, $\hat{y}_t$ is a random variable. Its variance reflects **model uncertainty** — how much the fitted value at $x_t$ would vary across different training sets:

  $$
  \mathbb{V}[\hat{y}_t] = \mathbb{V}[\hat{\beta}_0 + \hat{\beta}_1 x_t]
  $$

- **Variance of the actual response** (whether it is an observed value $y_t$ or a counterfactual value $y_*$ in the TBR setting):

  We model $y_t$ as:
  $$
  y_t = \hat{y}_t + \varepsilon_t
  $$
  where $\varepsilon_t \sim \mathcal{N}(0, \sigma^2)$ represents residual noise. Since $\hat{y}_t$ and $\varepsilon_t$ are independent, the total variance is:

  $$
  \mathbb{V}[y_t] = \mathbb{V}[\hat{y}_t + \varepsilon_t] = \mathbb{V}[\hat{y}_t] + \sigma^2
  $$

  This includes both the model uncertainty **and** the irreducible noise in the system, captured by $\sigma^2$, which accounts for natural variability in outcomes even if the model were known perfectly.

In summary, use $\mathbb{V}[\hat{y}_t]$ when constructing **confidence intervals for the model’s estimation at $x_t$**, and use $\mathbb{V}[y_t]$ when constructing **prediction intervals for new or counterfactual outcomes**.

## 📘 Credible Interval for the Posterior Distribution of $\Delta r(T)$

In the TBR framework, the **true cumulative causal effect** over the test period is defined as:

$$
\Delta r(T) = \sum_{t=1}^{T} (y_t - y_t^*)
$$

Since the counterfactual values $y_t^*$ are unobserved, we estimate them using a regression model trained on the pre-test period, resulting in predicted counterfactuals $\hat{y}_t^*$. The posterior distribution of the estimated causal effect is then based on the following model-derived quantity:

$$
\hat{\Delta r(T)} = \sum_{t=1}^{T} (y_t - \hat{y}_t^*)
$$

Under the assumptions of the TBR method, the **posterior distribution of $\Delta r(T)$** — given the data — is modeled as a Student’s $t$-distribution:

$$
\Delta r(T) \mid \text{data} \sim t_{\nu}(\hat{\Delta r(T)}, \sqrt{\mathbb{V}[\hat{\Delta r(T)}]})
$$

Where:
- $\mu \approx \hat{\Delta r(T)}$ is the posterior mean of the distribution (also the model estimate)
- $\sigma \approx \sqrt{\mathbb{V}[\hat{\Delta r(T)}]}$ is the posterior standard deviation
- $\nu$ is the degrees of freedom

### 🔹 Justification for Approximating $\mu$ and $\sigma$ in the Posterior Distribution of $\Delta r(T)$

The true expectation $\mathbb{E}[\Delta r(T)]$ and its variance $\mathbb{V}[\Delta r(T)]$ cannot be derived directly because the counterfactual outcomes $y_t^*$ are unobserved. So instead of $\Delta r(T) = \sum (y_t - y_t^*)$, we compute the estimator $\hat{\Delta r(T)} = \sum (y_t - \hat{y}_t^*)$, where $\hat{y}_t^*$ is the predicted counterfactual from a regression model trained on the pre-test period.

Under [standard assumptions](#-standard-assumptions-used-in-the-tbr-estimation-framework), the predicted counterfactuals $\hat{y}_t^* = \hat{\beta}_0 + \hat{\beta}_1 x_t$ are unbiased estimators of the conditional mean $\mathbb{E}[y_t^*]$. This is justified by:
$$
\mathbb{E}[\hat{y}_t^*] = \mathbb{E}[\hat{\beta}_0 + \hat{\beta}_1 x_t] = \beta_0 + \beta_1 x_t = \mathbb{E}[y_t^*]
$$

Taking the expectation of $\hat{\Delta r(T)} = \sum (y_t - \hat{y}_t^*)$ gives:
$$
\mathbb{E}[\hat{\Delta r(T)}] = \sum \left( \mathbb{E}[y_t] - \mathbb{E}[\hat{y}_t^*] \right) = \sum \left( \mathbb{E}[y_t] - \mathbb{E}[y_t^*] \right) = \mathbb{E}[\Delta r(T)]
$$

This shows that $\hat{\Delta r(T)}$ is an approximately unbiased estimator of $\mathbb{E}[\Delta r(T)]$, meaning that its expected value equals the target quantity under [standard assumptions](#-standard-assumptions-used-in-the-tbr-estimation-framework). Since we do not have access to $\mathbb{E}[\hat{\Delta r(T)}]$ or repeated realizations of the estimator, we use the single observed value $\hat{\Delta r(T)}$ as a stand-in for its expectation and, by extension, for $\mathbb{E}[\Delta r(T)]$. The estimated variance $\mathbb{V}[\hat{\Delta r(T)}]$ is then treated as a proxy for the unknown variance of the true effect, $\mathbb{V}[\Delta r(T)]$.


### Credible Interval Construction

Given this posterior distribution, the TBR method computes a symmetric credible interval at level $1 - \alpha$ using:

$$
[\texttt{lower}, \texttt{upper}] = \hat{\Delta r(T)} \pm t_{\alpha/2, \nu} \cdot \sqrt{\mathbb{V}[\hat{\Delta r(T)}]}
$$

Where:
- $t_{\alpha/2, \nu}$ is the quantile of the Student’s $t$-distribution with $\nu$ degrees of freedom
- $\alpha = 1 - \texttt{level}$, the probability outside the credible interval. For example, 0.2 for an 80% level of credibility

---

**Interpretation:**

This interval represents the range in which the **true cumulative causal effect** $\Delta r(T)$ lies with posterior probability $1 - \alpha$, conditional on the model and observed data. It reflects both:
- **Uncertainty from estimating the counterfactuals** $\hat{y}_t^*$
- **Residual noise** captured by the model’s standard deviation $\sigma$

The credible interval is centered at the mean of the distribution, $\hat{\Delta r(T)}$, and its width depends on the posterior uncertainty, $\sqrt{\mathbb{V}[\hat{\Delta r(T)}]}$ (`se`) and the chosen confidence level (`level`).


## 📘 Interpretation of TBR Summary Output

This section explains the meaning of each field returned in the TBR summary output of the GeoexperimentsResearch R package. Each parameter is described using the mathematical notation and assumptions developed in this document.

These parameters summarize the posterior distribution of the estimated cumulative causal effect and reflect both model-based uncertainty and underlying assumptions of the TBR framework.

👉 **Implementation Source:** The R function `summary.TBRAnalysisFitTbr1()` is the **definitive implementation** of how these summary statistics are computed. This implementation is the actual source used to generate values like `estimate`, `precision`, `se`, and credible intervals.  
You can view it here:  
[summary_tbranalysisfit_tbr1.R on GitHub](https://github.com/google/GeoexperimentsResearch/blob/master/R/summary_tbranalysisfit_tbr1.R)

### 🔹 `estimate`

The `estimate` field represents the **posterior estimate of the cumulative causal effect** over the entire test period. Mathematically, it corresponds to:

$$
\hat{\Delta r(T)} = \sum_{t=1}^{T} (y_t - \hat{y}_t^*)
$$

This quantity measures the total difference between the observed outcomes $y_t$ during the test period and their counterfactual predictions $\hat{y}_t^*$ — that is, the values the model estimates would have occurred **in the absence of treatment**.

- A **positive** estimate indicates an **increase** due to the treatment.
- A **negative** estimate indicates a **decrease** due to the treatment.

This is equivalent to the `cumdif` column over the test period.

### 🔹 `precision`

The `precision` field is defined in the R implementation as the **absolute value of the margin of error**, i.e., the half-width of the credible interval around the estimate:

$$
\texttt{precision} = | t_{\alpha/2, \nu} \cdot \texttt{se} | = \frac{1}{2} \left( \texttt{upper} - \texttt{lower} \right)
$$

Where:
- $\texttt{se} = \sqrt{\mathbb{V}[\hat{\Delta r(T)}]}$ is the posterior standard deviation
- $\alpha = 1 - \texttt{level}$, the total probability mass **outside** the credible interval (e.g., 0.2 for an 80% level)
- $t_{\alpha/2, \nu}$ is the two-tailed critical value from the Student’s $t$-distribution with $\nu$ degrees of freedom
- $\nu$ is the residual degrees of freedom, equal to the number of pre-test observations minus the number of model parameters.


### 🔹 `lower` and `upper`

The `lower` and `upper` fields define the bounds of the **posterior credible interval** for the true cumulative causal effect $\Delta r(T)$.

Let the posterior standard deviation be denoted by:

$$
\texttt{se} = \sqrt{ \mathbb{V}[\hat{\Delta r(T)}] }
$$

Then the credible interval is:

$$
[\texttt{lower}, \texttt{upper}] = \hat{\Delta r(T)} \pm t_{\alpha/2, \nu} \cdot \texttt{se}
$$

Where:
- $\hat{\Delta r(T)}$ is the posterior mean of the distribution (also the model estimate)
- $t_{\alpha/2, \nu}$ is the appropriate quantile from a Student’s $t$-distribution
- $\nu$ is the degrees of freedom
- $\alpha = 1 - \texttt{level}$, e.g., 0.2 for an 80% interval

The credible interval reflects the **posterior uncertainty** around the effect, assuming the TBR model and its variance estimates are correct.

### 🔹 `se`

The `se` field represents the **posterior standard deviation** of the estimated cumulative causal effect $\hat{\Delta r(T)}$ over the entire test period:

$$
\texttt{se} = \sqrt{ \mathbb{V}[\hat{\Delta r(T)}] }
$$

This value captures the posterior uncertainty of the cumulative treatment effect at the end of the test period.

In the R implementation, `se` is extracted as the final value of `y.pred.sd.cum` in the test period:

**From R package:**
```
# Final value of y.pred.sd.cum over the test period
se <- last(y.pred.sd.cum)
```
**On The Python code:**

Based on the Python variables, $\texttt{se}$ corresponds to the last value of `cumsd` where `period == 1` (i.e., the test period).

### 🔹 `level`

The `level` field specifies the **credibility level** used to construct the posterior credible interval for the estimated cumulative treatment effect.

It represents the posterior probability that the true cumulative treatment effect lies within the interval defined by `lower` and `upper`. For example, a level of 0.80 means there is an 80% probability — under the posterior distribution — that the effect lies between those bounds.

This value is set by the user when calling the `summary()` function in R, typically via the `level` argument:

**From R package:**
```
summary(obj_tbr, level = 0.80)
```

### 🔹 `thres`

The `thres` field specifies a **threshold value** used to compute the posterior probability that the true cumulative treatment effect $\Delta r(T)$ exceeds that threshold.

It is used to evaluate one-sided hypotheses such as $\Delta r(T) > \texttt{thres}$. In most cases, the threshold is set to 0 to assess whether the treatment had a positive effect.

This value is passed by the user when calling the `summary()` function in R, via the `threshold` argument:

**From R package:**
```
summary(obj_tbr, level = 0.80, threshold = 0)
```

**On the Python code:**

`thres` corresponds to the `ci_thres` argument passed into the summary() call.

### 🔹 `prob`

The `prob` field represents the **posterior probability** that the true cumulative treatment effect $\Delta r(T)$ exceeds the specified threshold $\texttt{thres}$:

$$
\texttt{prob} = P(\Delta r(T) > \texttt{thres} \mid \text{data})
$$

This value is computed using the posterior distribution of the effect, which is modeled as a Student’s $t$-distribution based on the estimated mean (`estimate`), standard deviation (`se`), and degrees of freedom (`t_dist_df`).

In most cases, `thres = 0`, so `prob` reflects the model’s belief that the treatment had a **positive** effect.

**From R package:**
```r
post.prob <- GetInfo(object, "tailprob")(threshold)
```

### 🔹 `model`

The `model` field specifies the name of the regression model used in the TBR analysis.

`tbr1` refers to the first implementation of the TBR model in the GeoexperimentsResearch package.

**From R code:**
```
obj_tbr <- DoTBRAnalysis(obj, model = "tbr1", response = kpi,
                         pretest.period = 0, intervention.period = 1,
                         cooldown.period = NULL, control.group = 1,
                         treatment.group = 2)
```

### 🔹 `alpha` and `beta`

The `alpha` and `beta` fields represent the estimated **regression coefficients** of the TBR model:

- `alpha` is the estimated **intercept** $\hat{\beta}_0$
- `beta` is the estimated **slope** $\hat{\beta}_1$, which multiplies the covariate $x_t$

These coefficients are fitted using ordinary least squares (OLS) on the **pre-test period only**, where the model is trained to predict the outcome $y_t$ as a linear function of the covariate $x_t$:

$$
\hat{y}_t = \hat{\beta}_0 + \hat{\beta}_1 x_t
$$

### 🔹 `alpha_beta_cov`

The `alpha_beta_cov` field represents the **covariance between the intercept and slope estimates** of the regression model:

$$
\texttt{alpha_beta_cov} = \text{Cov}(\hat{\beta}_0, \hat{\beta}_1)
$$

This value captures how uncertainty in the estimate of the intercept $\hat{\beta}_0$ is related to uncertainty in the estimate of the slope $\hat{\beta}_1$. It plays a role in the computation of the posterior variance of the cumulative predictions, $\mathbb{V} \left( \sum_{t=1}^{T} \hat{y}_t^* \right)$

### 🔹 `sigma`

The `sigma` field represents the **residual standard deviation** $s$ estimated from the regression model fitted to the pre-test period. `sigma`= $s$, where $s$ is defined earlier as:

$$
s^2 = \frac{1}{n - 2} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

This value estimates the standard deviation of the noise term $\varepsilon_t$ in the linear model:

$$
y_t = \beta_0 + \beta_1 x_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2)
$$

It accounts for the **residual noise** in the observed data and contributes directly to the posterior variance of the cumulative effect estimate.

**From R code:**
```
sigma <- tbr_lmpred[["residual.scale"]]  # Residual s.d.
```

### 🔹 `t_dist_df`

The `t_dist_df` field specifies the **degrees of freedom** $\nu$ used in the Student’s $t$-distribution that defines the posterior distribution of the cumulative treatment effect $\Delta r(T)$:

$$
\Delta r(T) \mid \text{data} \sim t_{\nu}\left(\hat{\Delta r(T)}, \sqrt{\mathbb{V}[\hat{\Delta r(T)}]} \right)
$$

This value is determined by the residual degrees of freedom from the regression model fitted to the pre-test period:

$$
\nu = n_{\text{pre}} - k
$$

Where:
- $n_{\text{pre}}$ is the number of observations in the pre-test period
- $k$ is the number of model parameters (typically 2: intercept and slope)

**Example:**  
If $n_{\text{pre}} = 90$ and $k = 2$, then $\nu = 88$.
---

## 📘 Posterior Variance and Credible Interval for a Subinterval of the Test Period

In many practical applications of the Time-Based Regression (TBR) method, it is useful to estimate the causal effect **over a subinterval** of the test period, rather than from the beginning. For instance, one may wish to exclude the initial days of the test period due to latency or stabilization effects. This section derives the posterior distribution of the cumulative causal effect from **day $i$ to day $j$**, where $1 \le i \le j \le T$, and $T$ is the number of days in the test period.

---

### 🔹 Definition of the Subinterval Effect

Let the cumulative causal effect over the subinterval $[i, j]$ be defined as:

$$
\hat{\Delta}_{[i \rightarrow j]} := \sum_{t=i}^{j} (y_t - \hat{y}_t^*)
$$

where:
- $y_t$ is the observed outcome in the treatment group at time $t$,
- $\hat{y}_t^*$ is the counterfactual prediction of what $y_t$ would have been in the absence of treatment,
- $y_t - \hat{y}_t^*$ is the pointwise causal effect at time $t$.

---

### 🔹 Posterior Distribution of $\Delta_{[i \rightarrow j]}$

The posterior distribution of $\Delta_{[i \rightarrow j]}$ is modeled as a Student's $t$-distribution:

$$
\Delta_{[i \rightarrow j]} \mid \text{data} \sim t_\nu \left( \hat{\Delta}_{[i \rightarrow j]},\ \sqrt{ \mathbb{V}[\Delta_{[i \rightarrow j]}]} \right)
$$

where:
- $\hat{\Delta}_{[i \rightarrow j]}$ is the posterior mean (point estimate),
- $\nu$ is the degrees of freedom estimated from the pretest period,
- $\mathbb{V}[\Delta_{[i \rightarrow j]}]$ is the posterior variance.

---

### 🔹 Posterior Variance

Assuming independence of daily residuals and model errors across time, the posterior variance is the sum of the pointwise variances:

$$
\mathbb{V}[\Delta_{[i \rightarrow j]}] = \sum_{t=i}^{j} \mathbb{V}[y_t - \hat{y}_t^*] = \sum_{t=i}^{j} \left( \mathbb{V}[\hat{y}_t^*] + \sigma^2 \right)
$$

Here:
- $\mathbb{V}[\hat{y}_t^*]$ is the posterior variance of the model prediction at time $t$,
- $\sigma^2$ is the residual variance estimated from the pretest period and assumed to be constant.

Thus:

$$
\mathbb{V}[\Delta_{[i \rightarrow j]}] = \sum_{t=i}^{j} \mathbb{V}[\hat{y}_t^*] + (j - i + 1) \cdot \sigma^2
$$

---

### 🔹 Posterior Standard Deviation and Precision

Let $n := j - i + 1$ be the number of days in the subinterval. Then:

- The posterior standard deviation (or standard error) is:

$$
\text{SE}_{[i \rightarrow j]} := \sqrt{ \mathbb{V}[\Delta_{[i \rightarrow j]}] } = \sqrt{ \sum_{t=i}^{j} \mathbb{V}[\hat{y}_t^*] + n \cdot \sigma^2 }
$$

- The precision, defined as the half-width of the posterior credible interval, is:

$$
\text{Precision}_{[i \rightarrow j]} := t_{\alpha/2, \nu} \cdot \text{SE}_{[i \rightarrow j]}
$$

Where $t_{\alpha/2, \nu}$ is the $(1 - \alpha/2)$ quantile of the Student’s $t$-distribution with $\nu$ degrees of freedom.

---

### 🔹 Credible Interval Bounds

The $100 \cdot (1 - \alpha)\%$ posterior credible interval for the subinterval causal effect is:

$$
\left[
\hat{\Delta}_{[i \rightarrow j]} - t_{\alpha/2, \nu} \cdot \text{SE}_{[i \rightarrow j]},\quad
\hat{\Delta}_{[i \rightarrow j]} + t_{\alpha/2, \nu} \cdot \text{SE}_{[i \rightarrow j]}
\right]
$$

Equivalently, using the outputs of the TBR analysis:

- Let the estimated causal effect be computed as:

$$
\hat{\Delta}_{[i \rightarrow j]} = \sum_{t=i}^{j} (y_t - \text{pred}_t)
$$

- Let `precision` denote the posterior half-width:

$$
\texttt{precision}_{[i \rightarrow j]} := t_{\alpha/2, \nu} \cdot \text{SE}_{[i \rightarrow j]}
$$

Then the credible interval becomes:

$$
\left[
\sum_{t=i}^{j} (y_t - \text{pred}_t) - \texttt{precision}_{[i \rightarrow j]},\quad
\sum_{t=i}^{j} (y_t - \text{pred}_t) + \texttt{precision}_{[i \rightarrow j]}
\right]
$$

This expression corresponds directly to the credible interval bounds as computed using the model output, and avoids recomputing the $t$-distribution multiplier or standard error explicitly.

---

### 🔍 Computing Subinterval Measures from TBR Output


### 🔹 Deriving $\text{SE}_{[i \rightarrow j]}$ from the TBR Output

To compute the posterior standard deviation $\text{SE}_{[i \rightarrow j]}$ from the TBR model outputs, we rely on two components provided by the TBR results:

#### 1. Pointwise Posterior Variance of the Model Prediction

For each day $t$ in the test period, the posterior variance of the model prediction $\mathbb{V}[\hat{y}_t^*]$ is reported as:

$$
\mathbb{V}[\hat{y}_t^*] = \texttt{estsd}_t^2
$$

Where:
- $\texttt{estsd}_t$ is the standard deviation of $\hat{y}_t^*$ due to model uncertainty alone,
- This value is provided in the TBR output table under the column labeled `estsd`, one value per test period day.

To obtain the total model-based variance over the interval $[i, j]$, sum these values squared:

$$
\sum_{t=i}^{j} \mathbb{V}[\hat{y}_t^*] = \sum_{t=i}^{j} \texttt{estsd}_t^2
$$

#### 2. Residual Variance

The residual variance $\sigma^2$ is constant across time and estimated from the pretest period. It accounts for the natural, irreducible noise in the observed data. This value is obtained from the scalar `sigma` in the TBR summary output.

To accumulate the residual variance across $n = j - i + 1$ days:

$$
\sum_{t=i}^{j} \sigma^2 = n \cdot \sigma^2
$$

#### 3. Combine to Obtain Posterior Variance and Standard Error

With both components available from TBR output, the posterior variance over the interval is:

$$
\mathbb{V}[\Delta_{[i \rightarrow j]}] = \sum_{t=i}^{j} \texttt{estsd}_t^2 + n \cdot \sigma^2
$$

Then the posterior standard deviation is:

$$
\text{SE}_{[i \rightarrow j]} = \sqrt{ \sum_{t=i}^{j} \texttt{estsd}_t^2 + n \cdot \sigma^2 }
$$

This value is then used to compute the credible interval as previously described.

### 🔹 Additive Decomposition of Subinterval Standard Error and Precision

When working with adjacent or nested subintervals of the test period, it is often useful to compute the posterior standard deviation or precision for a subinterval $[i, j]$ in terms of quantities already computed over longer intervals. This is valid under the assumption of independence across time and constant residual variance.

#### Posterior Standard Error (Squared)

Let $\text{SE}_{[a \rightarrow b]}$ denote the posterior standard deviation of the cumulative effect over the interval $[a, b]$. By definition:

$$
\text{SE}_{[a \rightarrow b]}^2 = \sum_{t=a}^{b} \mathbb{V}[\hat{y}_t^*] + (b - a + 1) \cdot \sigma^2
$$

Now consider the cumulative posterior standard error up to day $j$ and up to day $i-1$:

$$
\begin{aligned}
\text{SE}_{[1 \rightarrow j]}^2 &= \sum_{t=1}^{j} \mathbb{V}[\hat{y}_t^*] + j \cdot \sigma^2 \\\\
\text{SE}_{[1 \rightarrow (i - 1)]}^2 &= \sum_{t=1}^{i-1} \mathbb{V}[\hat{y}_t^*] + (i - 1) \cdot \sigma^2
\end{aligned}
$$

Subtracting these gives:

$$
\begin{aligned}
\text{SE}_{[i \rightarrow j]}^2 = \text{SE}_{[1 \rightarrow j]}^2 - \text{SE}_{[1 \rightarrow (i - 1)]}^2 \\\\
\text{SE}_{[i \rightarrow j]} = \sqrt{ \text{SE}_{[1 \rightarrow j]}^2 - \text{SE}_{[1 \rightarrow (i - 1)]}^2 }
\end{aligned}
$$

This identity follows from the linearity of variance over non-overlapping time periods and holds exactly under the TBR assumptions.

---

#### Posterior Precision (Squared)

Let $\text{Precision}_{[a \rightarrow b]}$ denote the half-width of the credible interval over the interval $[a, b]$:

$$
\text{Precision}_{[a \rightarrow b]} = t_{\alpha/2, \nu} \cdot \text{SE}_{[a \rightarrow b]}
$$

Assuming the $t$-quantile $t_{\alpha/2, \nu}$ is constant (i.e., $\nu$ is fixed across comparisons), we square both sides:

$$
\text{Precision}_{[a \rightarrow b]}^2 = t_{\alpha/2, \nu}^2 \cdot \text{SE}_{[a \rightarrow b]}^2
$$

Applying this to the identity derived above:

$$
\begin{aligned}
\text{Precision}_{[i \rightarrow j]}^2
&= t_{\alpha/2, \nu}^2 \cdot \text{SE}_{[i \rightarrow j]}^2 \\
&= t_{\alpha/2, \nu}^2 \cdot \left( \text{SE}_{[1 \rightarrow j]}^2 - \text{SE}_{[1 \rightarrow (i - 1)]}^2 \right) \\
&= \text{Precision}_{[1 \rightarrow j]}^2 - \text{Precision}_{[1 \rightarrow (i - 1)]}^2
\end{aligned}
$$

$$
\text{Precision}_{[i \rightarrow j]} = \sqrt{ \text{Precision}_{[1 \rightarrow j]}^2 - \text{Precision}_{[1 \rightarrow (i - 1)]}^2 }
$$

This decomposition enables precise and efficient computation of interval-specific uncertainties using already-accumulated posterior quantities.

### 🧾 Summary of Required Formulas

Let:
- $\hat{\Delta}_{[i \rightarrow j]} := \sum_{t=i}^{j} (y_t - \hat{y}_t^*)$
- $n := j - i + 1$
- $\mathbb{V}[\hat{y}_t^*]$ is the model variance at time $t$
- $\sigma^2$ is the residual variance from the pretest fit
- $\nu$ is the degrees of freedom from the regression
- $t_{\alpha/2, \nu}$ is the t-distribution quantile

Then:

1. **Point Estimate**:
   $$
   \hat{\Delta}_{[i \rightarrow j]} = \sum_{t=i}^{j} (y_t - \hat{y}_t^*)
   $$

2. **Posterior Variance**:
   $$
   \mathbb{V}[\Delta_{[i \rightarrow j]}] = \sum_{t=i}^{j} \mathbb{V}[\hat{y}_t^*] + n \cdot \sigma^2
   $$

3. **Posterior Standard Deviation**:
   $$
   \begin{aligned}
   \text{SE}_{[i \rightarrow j]} &= \sqrt{ \sum_{t=i}^{j} \mathbb{V}[\hat{y}_t^*] + n \cdot \sigma^2 } \\ \\
   \text{SE}_{[i \rightarrow j]} &= \sqrt{ \text{SE}_{[1 \rightarrow j]}^2 - \text{SE}_{[1 \rightarrow (i - 1)]}^2 }
   \end{aligned}
   $$

4. **Precision**:
   $$
   \begin{aligned}
   \text{Precision}_{[i \rightarrow j]} &= t_{\alpha/2, \nu} \cdot \text{SE}_{[i \rightarrow j]} \\ \\
   \text{Precision}_{[i \rightarrow j]} &= \sqrt{ \text{Precision}_{[1 \rightarrow j]}^2 - \text{Precision}_{[1 \rightarrow (i - 1)]}^2 }
   \end{aligned}
   $$
5. **Credible Interval**:
   $$
   \left[
   \hat{\Delta}_{[i \rightarrow j]} - \text{Precision}_{[i \rightarrow j]},\quad
   \hat{\Delta}_{[i \rightarrow j]} + \text{Precision}_{[i \rightarrow j]}
   \right]
   $$

---
## 📎 Appendix

### 🔹 Standard Assumptions Used in the TBR Estimation Framework

The derivations and justifications throughout this document rely on the following standard assumptions, which align with the classical linear regression model applied to the pre-test period:

1. **Linearity of the model**  
   The untreated outcome follows a linear relationship with the covariate:  
   $$
   y_t^* = \beta_0 + \beta_1 x_t + \varepsilon_t
   $$

2. **Exogeneity (zero-mean residuals)**  
   The error term has zero mean conditional on the covariates:  
   $$
   \mathbb{E}[\varepsilon_t \mid x_t] = 0
   $$
   This ensures that the regression estimators are unbiased:
   $$
   \mathbb{E}[\hat{\beta}_0] = \beta_0, \quad \mathbb{E}[\hat{\beta}_1] = \beta_1
   $$

3. **Homoscedasticity**  
   The variance of the residuals is constant across time:
   $$
   \mathbb{V}[\varepsilon_t \mid x_t] = \sigma^2
   $$


4. **Independence of residuals**  
   The residuals are assumed to be uncorrelated across time (i.e., no autocorrelation):  
   $$
   \text{Cov}(\varepsilon_t, \varepsilon_s) = 0 \quad \text{for all } t \ne s
   $$


5. **Model is trained only on pre-treatment data**  
   The regression model is fit exclusively on data from the pre-test period, ensuring that treatment effects do not bias parameter estimates.
