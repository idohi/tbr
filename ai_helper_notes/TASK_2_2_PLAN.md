# Task 2.2: Create Data Preprocessing and Cleaning Functions

## 📋 Task Overview

**Task ID**: 2.2
**Title**: Create data preprocessing and cleaning functions (extracted from functional implementation)
**Status**: 🔄 Not Started
**Dependencies**: ✅ 1.4 (functional code) + ✅ 2.1 (validation utilities) - **BOTH COMPLETE!**
**Objective**: Extract and enhance data preprocessing patterns from functional code into reusable utilities in `utils/preprocessing.py`

## 🔍 Current State Analysis

### ✅ Preprocessing Patterns Actually Present in Functional Code

Based on detailed analysis of `src/tbr/functional/tbr_functions.py`, we have identified **5 core preprocessing patterns** that can be safely extracted:

1. **`split_by_periods`** - Time series period splitting with boundary handling (lines 198-252)
2. **Array extraction** - Data extraction for regression (`x = learning_data[control_col].values`, line 313)
3. **DataFrame column assignments** - Period indicators and column mapping (lines 1221-1226, 1334-1350)
4. **Regression data preparation** - Adding constants for intercept (`sm.add_constant(x)`, line 329)
5. **Statistical calculations** - Mean calculations and basic statistics (`x_mean = np.mean(x)`, lines 106, 351)

**Note**: Other patterns are tightly coupled with TBR-specific logic and should remain in functional code.

### ✅ Current Dependencies Status

- **✅ 1.4 Complete**: Functional TBR implementation with all preprocessing patterns
- **✅ 2.1 Complete**: 17 validation functions ready for integration
- **✅ Infrastructure Ready**: Testing framework, CI/CD, code quality tools

## 🚀 Conservative Implementation Plan

### Phase 1: Extract Core Preprocessing Functions (ONLY)
**Goal**: Extract only the proven preprocessing patterns from functional code

#### 1.1 Time Series Period Splitting
- **Extract**: `split_by_periods()` function logic (lines 198-252)
- **Function**: `split_time_series_by_periods()`
- **Features**:
  - Direct extraction with minimal enhancement
  - Support for existing time column types
  - Maintain exact boundary handling logic
  - Preserve all validation and error handling

#### 1.2 Array Data Preparation
- **Extract**: Array extraction patterns (line 313: `x = learning_data[control_col].values`)
- **Function**: `extract_regression_arrays()`
- **Features**:
  - Safe array extraction with existing validation integration
  - Preserve data types and structure
  - Integration with validation utilities from Task 2.1

#### 1.3 DataFrame Column Processing
- **Extract**: Period indicator and column assignment patterns (lines 1221-1226, 1334-1350)
- **Function**: `assign_period_indicators()`
- **Features**:
  - Extract existing period assignment logic
  - Column mapping and renaming utilities
  - Preserve existing DataFrame structure

#### 1.4 Regression Data Preparation
- **Extract**: Statsmodels preparation (line 329: `sm.add_constant(x)`)
- **Function**: `prepare_regression_arrays()`
- **Features**:
  - Add constant for intercept
  - Basic data preparation for regression
  - Integration with existing statistical validation

#### 1.5 Statistical Utilities
- **Extract**: Basic statistical calculations (lines 106, 351: `np.mean(x)`)
- **Function**: `calculate_basic_statistics()`
- **Features**:
  - Mean, variance, and basic statistics
  - Integration with existing TBR calculations
  - Preserve mathematical accuracy

### Phase 2: Integration and Testing ONLY
**Goal**: Ensure extracted functions work with existing codebase

#### 2.1 Update Functional Code Integration
- Update `tbr_functions.py` to use new preprocessing utilities
- Maintain 100% backward compatibility
- Verify all 194 tests still pass

#### 2.2 Add Comprehensive Tests
- Unit tests for each extracted function
- Integration tests with validation utilities
- Performance tests to ensure no regression

## 🎯 Expected Deliverables

### 📁 File Structure After Completion
```
src/tbr/utils/
├── __init__.py           # Updated exports
├── constants.py          # Existing constants
├── exceptions.py         # Existing exceptions
├── validation.py         # Existing validation (17 functions)
└── preprocessing.py      # ✨ NEW with 5 functions
```

### 🔧 New Preprocessing Functions (5 total - Conservative)

#### Extracted from Functional Code
- `split_time_series_by_periods()` - Period splitting logic
- `extract_regression_arrays()` - Array extraction for regression
- `assign_period_indicators()` - DataFrame period assignment
- `prepare_regression_arrays()` - Statsmodels data preparation
- `calculate_basic_statistics()` - Mean and basic statistical calculations

## 🏆 Success Criteria (Conservative & Transparent)

1. **✅ All 5 preprocessing patterns extracted** from functional code (verified by line numbers)
2. **✅ 100% backward compatibility** maintained - all 194 tests must still pass
3. **✅ Integration verified** with existing 17 validation functions
4. **✅ Performance maintained** - no regression from original implementation
5. **✅ Professional documentation** with NumPy-style docstrings
6. **✅ Comprehensive test coverage** for each extracted function

## ⚡ Realistic Effort Estimate

- **Phase 1**: Extract 5 preprocessing functions (~90 minutes)
- **Phase 2**: Integration and testing (~90 minutes)
- **Total**: ~3 hours for conservative, reliable implementation

**Based on Task 2.1 experience**: 17 functions took significant effort, so 5 functions with proper testing is a realistic scope.

## 📚 Technical Details

### Function Signatures to Extract (Conservative)

```python
# 1. Time Series Period Splitting (from lines 198-252)
def split_time_series_by_periods(
    aggregated_data: pd.DataFrame,
    time_col: str,
    pretest_start: Union[pd.Timestamp, int, float],
    test_start: Union[pd.Timestamp, int, float],
    test_end: Union[pd.Timestamp, int, float],
    test_end_inclusive: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]

# 2. Array Extraction (from line 313)
def extract_regression_arrays(
    learning_data: pd.DataFrame,
    control_col: str,
    test_col: str,
) -> Tuple[np.ndarray, np.ndarray]

# 3. Period Indicators (from lines 1221-1226, 1334-1350)
def assign_period_indicators(
    data: pd.DataFrame,
    test_col: str,
    control_col: str,
    period_value: int
) -> pd.DataFrame

# 4. Regression Data Preparation (from line 329)
def prepare_regression_arrays(
    x: np.ndarray,
    add_constant: bool = True
) -> np.ndarray

# 5. Basic Statistics (from lines 106, 351)
def calculate_basic_statistics(
    x: np.ndarray
) -> Dict[str, float]
```

## 🔄 Implementation Steps (Conservative)

### Step 1: Extract Functions (Phase 1)
1. Create `utils/preprocessing.py` with proper imports and module structure
2. Extract `split_by_periods` logic exactly as-is (lines 198-252)
3. Extract array extraction patterns (line 313)
4. Extract period indicator assignment (lines 1221-1226, 1334-1350)
5. Extract regression data preparation (line 329)
6. Extract basic statistics calculations (lines 106, 351)

### Step 2: Integration and Testing (Phase 2)
1. Update `utils/__init__.py` exports for new preprocessing functions
2. Update functional code imports to use preprocessing utilities
3. Run all 194 tests to verify backward compatibility
4. Add unit tests for each extracted function (following Task 2.1 pattern)
5. Add integration tests with validation utilities
6. Performance testing to ensure no regression

## 📝 Notes

- **Maintain mathematical rigor**: All statistical preprocessing must preserve TBR methodology
- **Domain-agnostic design**: Functions should work across marketing, medical, economic domains
- **Performance focus**: No regression from current functional implementation speed
- **Professional standards**: Follow same code quality standards as validation utilities
- **Comprehensive testing**: Each function needs unit tests, integration tests, and edge cases
- **Documentation excellence**: NumPy-style docstrings with examples for each function

## 🎯 Ready to Proceed

This refined plan provides a **conservative, transparent roadmap** for implementing Task 2.2 following our scientific standards. The preprocessing utilities will enhance code organization while maintaining the proven functional implementation.

**Key Benefits:**
- **Transparency**: Only extract what actually exists and works
- **Reliability**: Conservative scope with realistic timeline
- **Maintainability**: Clear separation of concerns without over-engineering
- **Professional Quality**: Matches Task 2.1 validation utilities standards
- **Scientific Rigor**: Preserves mathematical accuracy and backward compatibility

---

**Created**: September 17, 2025
**Last Updated**: September 17, 2025
**Status**: Ready for Implementation
**Dependencies**: ✅ All prerequisites complete
