# TBR Mathematical Methodology

This document provides a comprehensive overview of the mathematical foundations underlying Time-Based Regression (TBR) analysis. It is intended for researchers, data scientists, and practitioners who want to understand the statistical methodology implemented in this package.

---

## Table of Contents

1. [Overview](#overview)
2. [Statistical Model](#statistical-model)
3. [Prediction and Uncertainty](#prediction-and-uncertainty)
4. [References](#references)

---

## Overview

### What is Time-Based Regression?

Time-Based Regression (TBR) is a statistical methodology for estimating causal treatment effects in before-after experimental designs. It addresses a fundamental question in causal inference: *What would have happened in the absence of the treatment?*

The core idea is straightforward:

1. **Pre-treatment period**: Establish a statistical relationship between a control group and a treatment group using linear regression
2. **Treatment period**: Use this relationship to predict *counterfactual* outcomes — what the treatment group would have experienced without the intervention
3. **Effect estimation**: Calculate the difference between observed outcomes and counterfactual predictions to estimate the treatment effect

TBR provides not just point estimates but complete statistical inference, including credible intervals and posterior probabilities for treatment effects.

### When to Use TBR

TBR is appropriate when you have:

- **Time series data** with measurements before and during a treatment period
- **Control and treatment groups** measured over the same time periods
- **A stable pre-treatment relationship** between control and treatment groups
- **No treatment contamination** in the control group during the test period

**Common applications include:**

- Marketing campaign effectiveness measurement
- Medical treatment outcome analysis
- Policy intervention evaluation
- A/B testing with time-based metrics
- Economic impact studies

TBR is particularly valuable when randomized controlled trials are impractical or when you need to measure cumulative effects over time.

### Key Advantages

**Statistical Rigor**
- Provides formal credible intervals with proper uncertainty quantification
- Accounts for both model uncertainty and residual variance
- Uses well-established regression theory with clear assumptions

**Interpretability**
- Results are expressed in the original units of measurement
- Cumulative effects show total impact over the test period
- Posterior probabilities give intuitive answers to business questions

**Flexibility**
- Works with any time granularity (daily, weekly, hourly, etc.)
- Supports subinterval analysis for custom time windows
- Handles various data types and domains

**Transparency**
- All assumptions are explicit and testable
- Diagnostic tools help validate model appropriateness
- Mathematical foundations are fully documented

---

## Statistical Model

### The Linear Regression Framework

TBR models the relationship between the treatment group metric ($y_t$) and the control group metric ($x_t$) using simple linear regression:

$$
y_t = \beta_0 + \beta_1 x_t + \varepsilon_t
$$

where:
- $y_t$ is the treatment group metric at time $t$
- $x_t$ is the control group metric at time $t$
- $\beta_0$ is the intercept (baseline offset)
- $\beta_1$ is the slope (scaling factor between groups)
- $\varepsilon_t$ is the random error term

The model is fitted using data from the **pre-treatment period only**, where no intervention has occurred. This ensures that the estimated relationship reflects the natural correlation between groups, uncontaminated by treatment effects.

### Model Assumptions

The TBR methodology relies on standard linear regression assumptions:

**1. Linearity**

The relationship between the control and treatment group metrics is linear. This means that changes in the control group metric produce proportional changes in the expected treatment group metric.

**2. Independence**

The error terms $\varepsilon_t$ are independent across time periods:

$$
\text{Cov}(\varepsilon_t, \varepsilon_s) = 0 \quad \text{for all } t \neq s
$$

**3. Homoscedasticity**

The error terms have constant variance across all time periods:

$$
\text{Var}(\varepsilon_t) = \sigma^2 \quad \text{for all } t
$$

**4. Normality**

The error terms follow a normal distribution:

$$
\varepsilon_t \sim \mathcal{N}(0, \sigma^2)
$$

**5. Exogeneity**

The error terms have zero mean conditional on the control group values:

$$
\mathbb{E}[\varepsilon_t \mid x_t] = 0
$$

This ensures that the regression estimators are unbiased.

**6. Stable Relationship**

The relationship established during the pre-treatment period continues to hold during the treatment period. This is the key assumption enabling counterfactual prediction.

### Parameter Estimation (OLS)

The regression coefficients are estimated using Ordinary Least Squares (OLS) on the pre-treatment data. Given $n$ observations in the pre-treatment period:

**Sample Means**

$$
\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i, \quad \bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i
$$

**Sum of Squared Deviations**

$$
S_{xx} = \sum_{i=1}^{n} (x_i - \bar{x})^2
$$

**Slope Estimator**

$$
\hat{\beta}_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{S_{xx}}
$$

**Intercept Estimator**

$$
\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}
$$

**Residual Variance Estimator**

The variance of the error term is estimated from the residuals:

$$
s^2 = \frac{1}{n - 2} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

where $\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$ is the fitted value. The denominator uses $n - 2$ degrees of freedom to provide an unbiased estimate of $\sigma^2$.

---

## Prediction and Uncertainty

Once the regression model has been fitted on the pre-treatment period, TBR uses it to generate **counterfactual predictions** — estimates of what the treatment group would have experienced in the absence of the intervention. Quantifying the uncertainty of these predictions is essential for valid statistical inference about treatment effects.

### Counterfactual Predictions

During the treatment period, the control group metric $x_t$ is observed but the treatment group is affected by the intervention. The counterfactual prediction for each time point $t$ in the treatment period is:

$$
\hat{y}_t^* = \hat{\beta}_0 + \hat{\beta}_1 x_t, \quad t \text{ in the test period}
$$

where:
- $\hat{y}_t^*$ is the predicted value the treatment group would have taken without treatment at time $t$
- $x_t$ is the observed control group metric at time $t$
- $\hat{\beta}_0$ and $\hat{\beta}_1$ are the regression coefficients estimated from the pre-treatment period

These predictions rely on the assumption that the relationship between the control and treatment groups, established during the pre-treatment period, continues to hold during the treatment period.

### Model Variance

The model variance captures the uncertainty in $\hat{y}_t^*$ that arises solely from estimating the regression coefficients $\hat{\beta}_0$ and $\hat{\beta}_1$. If the true coefficients were known, this component would be zero.

$$
\mathbb{V}[\hat{y}_t^*] = s^2 \left( \frac{1}{n} + \frac{(x_t - \bar{x})^2}{S_{xx}} \right)
$$

where:
- $s^2$ is the estimated residual variance from the pre-treatment period
- $n$ is the number of pre-treatment observations
- $\bar{x}$ is the mean of the control group metric in the pre-treatment period
- $S_{xx} = \sum_{i=1}^{n} (x_i - \bar{x})^2$ is the sum of squared deviations

The model variance is smallest when the control group value $x_t$ is close to the pre-treatment mean $\bar{x}$, and increases as $x_t$ moves further from $\bar{x}$. It also decreases with larger pre-treatment sample size $n$.

### Prediction Variance

The prediction variance captures the **total** uncertainty in a counterfactual observation $y_t^*$, combining both the model uncertainty and the irreducible residual noise:

$$
y_t^* = \hat{y}_t^* + \varepsilon_t^*, \quad \varepsilon_t^* \sim \mathcal{N}(0, \sigma^2)
$$

Since $\hat{y}_t^*$ and $\varepsilon_t^*$ are independent:

$$
\mathbb{V}[y_t^*] = \mathbb{V}[\hat{y}_t^*] + \sigma^2
$$

Substituting the model variance and replacing $\sigma^2$ with its estimator $s^2$:

$$
\mathbb{V}[y_t^*] = s^2 \left( 1 + \frac{1}{n} + \frac{(x_t - \bar{x})^2}{S_{xx}} \right)
$$

The prediction standard deviation is then:

$$
\sqrt{\mathbb{V}[y_t^*]} = s \cdot \sqrt{1 + \frac{1}{n} + \frac{(x_t - \bar{x})^2}{S_{xx}}}
$$

### Distinguishing Model Variance and Prediction Variance

These two quantities serve different purposes in the TBR framework:

| Quantity | Formula | Source of uncertainty | Role in TBR |
|----------|---------|----------------------|-------------|
| Model variance $\mathbb{V}[\hat{y}_t^*]$ | $s^2 \left( \frac{1}{n} + \frac{(x_t - \bar{x})^2}{S_{xx}} \right)$ | Estimation of regression coefficients | Cumulative effect variance and credible intervals |
| Prediction variance $\mathbb{V}[y_t^*]$ | $s^2 \left( 1 + \frac{1}{n} + \frac{(x_t - \bar{x})^2}{S_{xx}} \right)$ | Coefficient estimation **and** residual noise | Pointwise prediction intervals for counterfactual outcomes |

The prediction variance is always larger than the model variance by exactly $s^2$ (the residual variance term). In practice:

- **Model variance** ($\mathbb{V}[\hat{y}_t^*]$) is used when computing the cumulative effect variance and credible intervals for the treatment effect
- **Prediction variance** ($\mathbb{V}[y_t^*]$) is used for pointwise prediction intervals around individual counterfactual values

Both quantities increase as the control group value $x_t$ moves further from $\bar{x}$, reflecting greater extrapolation uncertainty.

---

## References

Kerman, J., Wang, P., & Vaver, J. (2017). *Estimating Ad Effectiveness using Geo Experiments in a Time-Based Regression Framework*. Technical Report, Google, Inc. [PDF](https://research.google/pubs/pub45950/)

---
