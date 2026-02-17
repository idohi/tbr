# TBR Mathematical Methodology

This document provides a comprehensive overview of the mathematical foundations underlying Time-Based Regression (TBR) analysis. It is intended for researchers, data scientists, and practitioners who want to understand the statistical methodology implemented in this package.

---

## Table of Contents

1. [Overview](#overview)
2. [Statistical Model](#statistical-model)
3. [Prediction and Uncertainty](#prediction-and-uncertainty)
4. [Treatment Effect Estimation](#treatment-effect-estimation)
5. [Statistical Inference](#statistical-inference)
6. [Subinterval Analysis](#subinterval-analysis)
7. [Model Diagnostics](#model-diagnostics)
8. [Assumptions and Limitations](#assumptions-and-limitations)
9. [Notation Reference](#notation-reference)
10. [References](#references)

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

### Counterfactual Predictions

<!-- TODO: Task A4 - Explain what counterfactual predictions are -->

### Prediction Variance

<!-- TODO: Task A4 - Present and explain V[y*] formula -->

### Model Variance

<!-- TODO: Task A4 - Present and explain V[ŷ*] formula -->

### Distinguishing Prediction vs Model Variance

<!-- TODO: Task A4 - Clarify when to use each -->

---

## Treatment Effect Estimation

### Daily Treatment Effect

<!-- TODO: Task A5 - Define daily lift (y_t - ŷ_t*) -->

### Cumulative Treatment Effect

<!-- TODO: Task A5 - Define Δr(T) = Σ(y_t - ŷ_t*) -->

### Posterior Variance of Cumulative Effect

<!-- TODO: Task A5 - Present V[Δr(T)] = T·σ² + T²·v formula -->

---

## Statistical Inference

### The t-Distribution Framework

<!-- TODO: Task A6 - Explain why t-distribution is used -->

### Credible Intervals

<!-- TODO: Task A6 - Present credible interval formula -->

### Posterior Probability

<!-- TODO: Task A6 - Explain P(Δr(T) > threshold | data) -->

### Degrees of Freedom

<!-- TODO: Task A6 - Explain how df is determined -->

---

## Subinterval Analysis

### Custom Time Window Analysis

<!-- TODO: Task A7 - Explain analyzing subsets of test period -->

### Variance Calculation for Subintervals

<!-- TODO: Task A7 - Present subinterval variance formula -->

---

## Model Diagnostics

### Residual Analysis

<!-- TODO: Task A8 - Describe residual-based diagnostics -->

### Assumption Checking

<!-- TODO: Task A8 - List diagnostic tests available -->

### Goodness-of-Fit Metrics

<!-- TODO: Task A8 - Describe R², AIC, BIC, etc. -->

---

## Assumptions and Limitations

### When TBR is Appropriate

<!-- TODO: Task A9 - Describe ideal scenarios for TBR -->

### Key Assumptions

<!-- TODO: Task A9 - List the critical assumptions -->

### Potential Violations and Remedies

<!-- TODO: Task A9 - Discuss what happens when assumptions are violated -->

---

## Notation Reference

### Variable Mapping

<!-- TODO: Task A10 - Create table mapping symbols to Python variables -->

| Symbol | Description | Python Variable |
|--------|-------------|-----------------|
| | | |

---

## References

### Primary Reference

<!-- TODO: Task A11 - Cite Kerman, Wang & Vaver (2017) -->

### Additional Reading

<!-- TODO: Task A11 - Add any supplementary references -->

---
