# Task 2.1: Implement Comprehensive Input Validation Utilities

## 📋 Task Overview

**Task ID**: 2.1
**Title**: Implement comprehensive input validation utilities based on functional code patterns
**Status**: 🔄 Not Started
**Dependencies**: ✅ 1.4 (functional code) + ✅ 1.5 (constants) - **BOTH COMPLETE!**
**Objective**: Extract validation patterns from functional code into reusable utilities in `utils/validation.py`

## 🔍 Current State Analysis

### ✅ Validation Functions Already in Functional Code

Based on analysis of `src/tbr/functional/tbr_functions.py`, we have **8 validation functions** that need to be extracted:

1. **`validate_required_columns`** - Check DataFrame has required columns
2. **`validate_time_column_type`** - Validate time column dtypes (datetime64, int64, float64)
3. **`validate_no_nulls`** - Check for null values in specified columns
4. **`validate_time_boundaries_type`** - Validate time boundary type consistency
5. **`validate_time_periods`** - Validate time period logical relationships
6. **`validate_metric_columns`** - Validate control/test columns are numeric
7. **`validate_period_data`** - Validate period splitting results
8. **`validate_learning_set`** - Validate regression training data

### ✅ Already Exist in Utils

- **`validate_array_not_empty`** - Array emptiness validation
- **`validate_sample_size`** - Statistical sample size validation

## 🚀 Detailed Implementation Plan

### Phase 1: Extract & Organize Validation Functions
**Goal**: Move validation functions from functional code to utils for reusability

#### 1.1 Move Core DataFrame Validation Functions
- Extract `validate_required_columns()`
- Extract `validate_no_nulls()`
- Extract `validate_metric_columns()`

#### 1.2 Move Time-Related Validation Functions
- Extract `validate_time_column_type()`
- Extract `validate_time_boundaries_type()`
- Extract `validate_time_periods()`

#### 1.3 Move Data Quality Validation Functions
- Extract `validate_period_data()`
- Extract `validate_learning_set()`

### Phase 2: Enhance & Extend Validation Utilities
**Goal**: Add comprehensive validation patterns for scientific computing

#### 2.1 Statistical Parameter Validation
- Probability/confidence level validation (0 ≤ level ≤ 1)
- Threshold parameter validation
- Degrees of freedom validation
- Variance parameter validation (non-negative)

#### 2.2 Data Quality Validation
- Missing data pattern detection
- Outlier detection utilities
- Data type consistency checking
- Range validation utilities

#### 2.3 TBR-Specific Validation
- Time series continuity validation
- Treatment/control group balance validation
- Minimum period length validation

### Phase 3: Professional Integration
**Goal**: Ensure seamless integration with existing codebase

#### 3.1 Update Imports
- Update `tbr_functions.py` to import from utils
- Update `utils/__init__.py` exports
- Ensure backward compatibility

#### 3.2 Comprehensive Testing
- Add tests for all extracted functions
- Add tests for new validation utilities
- Integration tests with functional code

#### 3.3 Documentation
- Professional docstrings for all functions
- Usage examples and edge cases
- Type hints and error handling

## 🎯 Expected Deliverables

### 📁 File Structure After Completion
```
src/tbr/utils/
├── __init__.py           # Updated exports
├── constants.py          # Existing constants
├── exceptions.py         # Existing exceptions
└── validation.py         # ✨ ENHANCED with 15+ functions
```

### 🔧 New Validation Functions (15+ total)

#### Core Data Validation
- `validate_required_columns()`
- `validate_no_nulls()`
- `validate_metric_columns()`
- `validate_dataframe_not_empty()`
- `validate_column_types()`

#### Time & Period Validation
- `validate_time_column_type()`
- `validate_time_boundaries_type()`
- `validate_time_periods()`
- `validate_period_data()`
- `validate_time_series_continuity()`

#### Statistical Validation
- `validate_confidence_level()`
- `validate_threshold_parameter()`
- `validate_degrees_freedom()`
- `validate_variance_parameters()`
- `validate_learning_set()`

#### Array & Sample Validation
- `validate_array_not_empty()` (existing)
- `validate_sample_size()` (existing)

## 🏆 Success Criteria

1. **✅ All 8 validation functions extracted** from functional code
2. **✅ 7+ new validation utilities** added for comprehensive coverage
3. **✅ 100% backward compatibility** maintained
4. **✅ Comprehensive test coverage** for all validation functions
5. **✅ Professional documentation** with examples and edge cases
6. **✅ Integration verified** with existing functional code

## ⚡ Estimated Effort

- **Phase 1**: Extract functions (~30 minutes)
- **Phase 2**: Enhance utilities (~45 minutes)
- **Phase 3**: Integration & testing (~30 minutes)
- **Total**: ~2 hours for comprehensive professional implementation

## 📚 Technical Details

### Function Signatures to Extract

```python
# Core DataFrame Validation
def validate_required_columns(df: pd.DataFrame, required_cols: List[str], df_name: str) -> None
def validate_no_nulls(df: pd.DataFrame, cols: List[str], df_name: str) -> None
def validate_metric_columns(data: pd.DataFrame, control_col: str, test_col: str) -> None

# Time-Related Validation
def validate_time_column_type(data: pd.DataFrame, time_col: str, df_name: str = "data") -> None
def validate_time_boundaries_type(pretest_start: Union[pd.Timestamp, int, float],
                                  test_start: Union[pd.Timestamp, int, float],
                                  test_end: Union[pd.Timestamp, int, float],
                                  time_column_dtype: np.dtype) -> None
def validate_time_periods(pretest_start: Union[pd.Timestamp, int, float],
                         test_start: Union[pd.Timestamp, int, float],
                         test_end: Union[pd.Timestamp, int, float],
                         test_end_inclusive: bool = False) -> None

# Data Quality Validation
def validate_period_data(pretest_data: pd.DataFrame, test_data: pd.DataFrame) -> None
def validate_learning_set(learning_df: pd.DataFrame, control_col: str, test_col: str) -> None
```

### New Functions to Implement

```python
# Statistical Parameter Validation
def validate_confidence_level(level: float, param_name: str = "confidence level") -> None
def validate_threshold_parameter(threshold: float, param_name: str = "threshold") -> None
def validate_degrees_freedom(df: int, param_name: str = "degrees of freedom") -> None
def validate_variance_parameters(**variances) -> None

# Enhanced Data Validation
def validate_dataframe_not_empty(df: pd.DataFrame, df_name: str) -> None
def validate_column_types(df: pd.DataFrame, column_types: Dict[str, str]) -> None
def validate_time_series_continuity(df: pd.DataFrame, time_col: str) -> None
```

## 🔄 Implementation Steps

### Step 1: Backup and Prepare
1. Create backup of current `validation.py`
2. Identify all validation functions in `tbr_functions.py`
3. Plan import updates

### Step 2: Extract Functions
1. Copy validation functions from `tbr_functions.py`
2. Add to `utils/validation.py` with proper organization
3. Update imports in `utils/validation.py`

### Step 3: Add New Utilities
1. Implement statistical parameter validation
2. Add enhanced data quality checks
3. Create TBR-specific validation utilities

### Step 4: Update Imports
1. Update `tbr_functions.py` imports
2. Update `utils/__init__.py` exports
3. Verify backward compatibility

### Step 5: Testing & Documentation
1. Add comprehensive tests
2. Update documentation
3. Verify integration

## 📝 Notes

- Maintain backward compatibility at all costs
- Follow existing code style and documentation patterns
- Ensure all validation functions have comprehensive error messages
- Add type hints for all parameters
- Include usage examples in docstrings

## 🎯 Ready to Proceed

This plan provides a comprehensive roadmap for implementing Task 2.1 with professional standards and complete integration with the existing TBR codebase.

---

**Created**: September 15, 2025
**Last Updated**: September 15, 2025
**Status**: Ready for Implementation
