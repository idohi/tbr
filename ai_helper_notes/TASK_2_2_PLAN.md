# Task 2.2: Create Data Preprocessing and Cleaning Functions

## 📋 Task Overview

**Task ID**: 2.2
**Title**: Create data preprocessing and cleaning functions (extracted from functional implementation)
**Status**: 🔄 Not Started
**Dependencies**: ✅ 1.4 (functional code) + ✅ 2.1 (validation utilities) - **BOTH COMPLETE!**
**Objective**: Extract and enhance data preprocessing patterns from functional code into reusable utilities in `utils/preprocessing.py`

## 🔍 Current State Analysis

### ✅ Preprocessing Patterns Identified in Functional Code

Based on analysis of `src/tbr/functional/tbr_functions.py`, we have identified **8 key preprocessing patterns** that need to be extracted:

1. **`split_by_periods`** - Time series period splitting with boundary handling
2. **Array preparation** - Data extraction and validation for regression (`x = learning_data[control_col].values`)
3. **Statistical preprocessing** - Variance checks, mean calculations, finite value validation
4. **DataFrame transformations** - Column assignments, period indicators, data structure reorganization
5. **Data quality checks** - Constant value detection, null validation, sufficiency testing
6. **Regression data preparation** - Adding constants for intercept (`sm.add_constant(x)`)
7. **Period data processing** - Baseline, pretest, test, cooldown data handling
8. **Statistical validation** - Model parameter validation, coefficient checks

### ✅ Current Dependencies Status

- **✅ 1.4 Complete**: Functional TBR implementation with all preprocessing patterns
- **✅ 2.1 Complete**: 17 validation functions ready for integration
- **✅ Infrastructure Ready**: Testing framework, CI/CD, code quality tools

## 🚀 Detailed Implementation Plan

### Phase 1: Extract Core Time Series Preprocessing Functions
**Goal**: Move time series specific preprocessing from functional code to utils

#### 1.1 Time Series Period Splitting
- **Extract**: `split_by_periods()` function logic
- **Enhance**: Add flexible boundary handling and custom period definitions
- **Function**: `split_time_series_by_periods()`
- **Features**:
  - Support for multiple time column types (datetime64, int64, float64)
  - Flexible boundary inclusion/exclusion logic
  - Custom period naming and indicators
  - Empty period handling and validation

#### 1.2 Time Series Data Preparation
- **Extract**: Array extraction patterns from regression fitting
- **Function**: `prepare_time_series_arrays()`
- **Features**:
  - Safe array extraction with validation
  - Type conversion and normalization
  - Missing value detection and reporting
  - Statistical sufficiency validation

#### 1.3 Time Series Quality Assessment
- **Extract**: Data quality patterns from functional validation
- **Function**: `assess_time_series_quality()`
- **Features**:
  - Continuity validation (gaps, duplicates)
  - Distribution assessment (outliers, skewness)
  - Statistical properties validation
  - Data sufficiency reporting

### Phase 2: Statistical Data Preprocessing Functions
**Goal**: Add enhanced statistical preprocessing capabilities

#### 2.1 Regression Data Preparation
- **Extract**: Statsmodels preparation patterns
- **Function**: `prepare_regression_data()`
- **Features**:
  - Add constant for intercept (extracted from `sm.add_constant(x)`)
  - Validate regression assumptions
  - Handle multicollinearity detection
  - Prepare data for various regression types

#### 2.2 Statistical Validation and Cleaning
- **Extract**: Statistical validation patterns
- **Functions**:
  - `validate_regression_assumptions()`
  - `detect_statistical_outliers()`
  - `assess_data_distributions()`
- **Features**:
  - Linearity assumption testing
  - Homoscedasticity validation
  - Independence testing
  - Normality assessment

#### 2.3 Data Quality Enhancement
- **New**: Advanced data cleaning capabilities
- **Functions**:
  - `handle_missing_values()`
  - `normalize_time_series_data()`
  - `standardize_data_formats()`
- **Features**:
  - Multiple imputation strategies
  - Robust normalization methods
  - Format standardization across domains

### Phase 3: DataFrame Processing and Transformation
**Goal**: Extract DataFrame manipulation patterns for reusability

#### 3.1 DataFrame Structure Processing
- **Extract**: DataFrame transformation patterns from `_create_tbr_dataframe()`
- **Functions**:
  - `assign_period_indicators()`
  - `restructure_analysis_dataframe()`
  - `combine_period_dataframes()`
- **Features**:
  - Flexible period coding systems
  - Column assignment and renaming
  - Multi-period data combination
  - Index management and sorting

#### 3.2 Column Processing and Validation
- **Extract**: Column processing patterns
- **Functions**:
  - `validate_column_data_types()`
  - `standardize_column_names()`
  - `create_derived_columns()`
- **Features**:
  - Type validation and conversion
  - Naming convention enforcement
  - Derived metric calculations
  - Column dependency validation

### Phase 4: Domain-Agnostic Data Preprocessing
**Goal**: Add preprocessing utilities that work across all domains

#### 4.1 Universal Data Cleaning
- **New**: Domain-independent cleaning functions
- **Functions**:
  - `clean_time_series_data()`
  - `validate_experimental_design()`
  - `prepare_control_test_data()`
- **Features**:
  - Universal outlier detection
  - Experimental design validation
  - Control/test group balance checking
  - Cross-domain compatibility

#### 4.2 Advanced Preprocessing Pipeline
- **New**: Comprehensive preprocessing pipeline
- **Functions**:
  - `create_preprocessing_pipeline()`
  - `validate_preprocessing_results()`
  - `generate_preprocessing_report()`
- **Features**:
  - Configurable preprocessing steps
  - Pipeline validation and testing
  - Comprehensive reporting
  - Performance monitoring

## 🎯 Expected Deliverables

### 📁 File Structure After Completion
```
src/tbr/utils/
├── __init__.py           # Updated exports
├── constants.py          # Existing constants
├── exceptions.py         # Existing exceptions
├── validation.py         # Existing validation (17 functions)
└── preprocessing.py      # ✨ NEW with 15+ functions
```

### 🔧 New Preprocessing Functions (15+ total)

#### Core Time Series Processing
- `split_time_series_by_periods()`
- `prepare_time_series_arrays()`
- `assess_time_series_quality()`
- `validate_time_series_continuity()`
- `detect_time_series_gaps()`

#### Statistical Data Preparation
- `prepare_regression_data()`
- `validate_regression_assumptions()`
- `detect_statistical_outliers()`
- `assess_data_distributions()`
- `normalize_statistical_data()`

#### DataFrame Processing
- `assign_period_indicators()`
- `restructure_analysis_dataframe()`
- `combine_period_dataframes()`
- `validate_column_data_types()`
- `standardize_column_names()`

#### Advanced Preprocessing
- `clean_time_series_data()`
- `create_preprocessing_pipeline()`
- `generate_preprocessing_report()`

## 🏆 Success Criteria

1. **✅ All 8 preprocessing patterns extracted** from functional code
2. **✅ 7+ new preprocessing utilities** added for comprehensive coverage
3. **✅ 100% backward compatibility** maintained with functional code
4. **✅ Comprehensive test coverage** for all preprocessing functions
5. **✅ Professional documentation** with examples and domain-agnostic use cases
6. **✅ Integration verified** with existing functional and validation code
7. **✅ Performance maintained** - no regression from original implementation

## ⚡ Estimated Effort

- **Phase 1**: Extract time series functions (~45 minutes)
- **Phase 2**: Statistical preprocessing (~60 minutes)
- **Phase 3**: DataFrame processing (~45 minutes)
- **Phase 4**: Advanced preprocessing (~45 minutes)
- **Integration & Testing**: (~45 minutes)
- **Total**: ~4 hours for comprehensive professional implementation

## 📚 Technical Details

### Function Signatures to Extract

```python
# Core Time Series Processing
def split_time_series_by_periods(
    data: pd.DataFrame,
    time_col: str,
    periods: Dict[str, Tuple[Union[pd.Timestamp, int, float], Union[pd.Timestamp, int, float]]],
    boundary_handling: str = "exclusive"
) -> Dict[str, pd.DataFrame]

def prepare_time_series_arrays(
    data: pd.DataFrame,
    control_col: str,
    test_col: str,
    validate_assumptions: bool = True
) -> Tuple[np.ndarray, np.ndarray]

def assess_time_series_quality(
    data: pd.DataFrame,
    time_col: str,
    metric_cols: List[str]
) -> Dict[str, Any]

# Statistical Data Preparation
def prepare_regression_data(
    x: np.ndarray,
    add_constant: bool = True,
    validate_assumptions: bool = True
) -> np.ndarray

def validate_regression_assumptions(
    x: np.ndarray,
    y: np.ndarray,
    assumption_tests: List[str] = None
) -> Dict[str, bool]

def detect_statistical_outliers(
    data: np.ndarray,
    method: str = "iqr",
    threshold: float = 1.5
) -> np.ndarray

# DataFrame Processing
def assign_period_indicators(
    dataframes: Dict[str, pd.DataFrame],
    period_mapping: Dict[str, int]
) -> Dict[str, pd.DataFrame]

def restructure_analysis_dataframe(
    data: pd.DataFrame,
    time_col: str,
    value_cols: List[str],
    period_col: str = "period"
) -> pd.DataFrame
```

### New Functions to Implement

```python
# Advanced Preprocessing
def clean_time_series_data(
    data: pd.DataFrame,
    time_col: str,
    metric_cols: List[str],
    cleaning_strategy: Dict[str, Any] = None
) -> pd.DataFrame

def create_preprocessing_pipeline(
    steps: List[Tuple[str, callable]],
    validate_steps: bool = True
) -> callable

def generate_preprocessing_report(
    original_data: pd.DataFrame,
    processed_data: pd.DataFrame,
    preprocessing_steps: List[str]
) -> Dict[str, Any]

# Data Quality Enhancement
def handle_missing_values(
    data: pd.DataFrame,
    strategy: str = "interpolate",
    max_missing_ratio: float = 0.1
) -> pd.DataFrame

def normalize_time_series_data(
    data: pd.DataFrame,
    method: str = "z_score",
    columns: List[str] = None
) -> pd.DataFrame
```

## 🔄 Implementation Steps

### Step 1: Setup and Analysis
1. Create `utils/preprocessing.py` with proper imports
2. Analyze functional code preprocessing patterns in detail
3. Plan integration points with validation utilities

### Step 2: Extract Core Functions (Phase 1)
1. Extract `split_by_periods` logic and enhance
2. Extract array preparation patterns
3. Extract data quality assessment patterns
4. Add comprehensive tests for extracted functions

### Step 3: Add Statistical Functions (Phase 2)
1. Implement regression data preparation utilities
2. Add statistical validation and cleaning functions
3. Create data quality enhancement utilities
4. Test integration with existing validation functions

### Step 4: DataFrame Processing (Phase 3)
1. Extract DataFrame transformation patterns
2. Add column processing and validation utilities
3. Create period indicator assignment functions
4. Validate DataFrame structure consistency

### Step 5: Advanced Features (Phase 4)
1. Implement domain-agnostic cleaning functions
2. Create preprocessing pipeline utilities
3. Add comprehensive reporting capabilities
4. Performance testing and optimization

### Step 6: Integration and Testing
1. Update functional code to use new preprocessing utilities
2. Update `utils/__init__.py` exports
3. Run comprehensive integration tests
4. Verify backward compatibility and performance

## 📝 Notes

- **Maintain mathematical rigor**: All statistical preprocessing must preserve TBR methodology
- **Domain-agnostic design**: Functions should work across marketing, medical, economic domains
- **Performance focus**: No regression from current functional implementation speed
- **Professional standards**: Follow same code quality standards as validation utilities
- **Comprehensive testing**: Each function needs unit tests, integration tests, and edge cases
- **Documentation excellence**: NumPy-style docstrings with examples for each function

## 🎯 Ready to Proceed

This plan provides a comprehensive roadmap for implementing Task 2.2 with professional standards. The preprocessing utilities will enhance the package's usability while maintaining the proven functional implementation as the foundation.

**Key Benefits:**
- **Reusability**: Preprocessing functions can be used independently
- **Maintainability**: Clear separation of concerns
- **Extensibility**: Easy to add new preprocessing capabilities
- **Professional Quality**: Matches validation utilities standards
- **Domain Agnostic**: Works across all TBR applications

---

**Created**: September 17, 2025
**Last Updated**: September 17, 2025
**Status**: Ready for Implementation
**Dependencies**: ✅ All prerequisites complete
