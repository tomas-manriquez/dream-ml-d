# Research Documentation Refinement Tasks - DREAM-ML Time Series

**Date**: 2025-11-17
**Status**: In Progress
**Purpose**: Track remaining refinements for time series training research documents

---

## Context

These tasks complete the refinement of 6 research documents to align with feature requirements:
- Univariate time series ONLY (1 target for all models)
- Date column strictly required
- Feature engineering in Data Encoding step (not Training)
- Auto-selection project rule
- LSTM validation correctly blocks raw target (not engineered lag features)

---

## CRITICAL Tasks (Blocking Correct Understanding)

###  Task 1: Complete System Constraints Sections
**Status**: PARTIALLY COMPLETE
**Completed**: [backend-analysis.md](2025-11-14_backend-analysis.md)
**Remaining Docs**:
- [ ] frontend-analysis.md
- [ ] validation-logic.md
- [ ] variable-selection-research.md
- [ ] recommendations.md

**Action**: Add "System-Level Constraints" section after Executive Summary:
```markdown
## System-Level Constraints

### Univariate Time Series Only
- ALL models (ARIMA, XGBoost, LSTM) support exactly 1 target variable
- Feature variables are inputs (X), not additional targets (Y)
- "Multivariate inputs" means multiple features, NOT multiple target variables
- This is a fundamental constraint of the system

### Date Column Requirement
- Strictly required for all algorithms (no implicit integer index)
- Must be parseable to datetime format
- Used for temporal ordering and chronological splits

### Feature Engineering Workflow (2-Step Process)
1. Data Encoding: Configure lags → backend generates → saves CSV
2. Training: Load encoded CSV → select from ALL columns (original + engineered)

### Auto-Selection Project Rule
- ARIMA/XGBoost: Auto-select all except target/date
- LSTM: NO auto-selection (manual checkboxes)
- Minimum 0 features allowed (univariate mode)

### Validation Rules
- Target: 1 (required, radio button)
- Features: 0+ (optional, checkboxes)
- Date: 1 (required, radio button)
```

**Priority**: CRITICAL
**Estimated Time**: 20 minutes per doc (1.3 hours total)

---

### ✅ Task 2: Clarify "Multivariate" Terminology
**Status**: COMPLETE
**Completed**: [backend-analysis.md](2025-11-14_backend-analysis.md) (line 16-19)

**Remaining**:
- [ ] variable-selection-research.md (add terminology clarification section)

**Action for variable-selection-research.md**:
Add after line 15:
```markdown
## Terminology Clarification

**CRITICAL**: This document uses "multivariate" to mean "multiple input features"

**Univariate Time Series (System Constraint)**:
- 1 target variable (Y) - what we predict
- 0+ feature variables (X) - what we use to predict
- This is a UNIVARIATE time series (single target)

**Multivariate Input Features** (NOT multivariate time series):
- LSTM with [Sales, Temperature, Humidity] in sequences
- Still predicts ONLY Sales (1 target)
- "Multivariate LSTM" means "multiple X variables", NOT "multiple Y variables"

**Multivariate Time Series** (NOT SUPPORTED):
- Multiple targets: predict [Sales, Inventory, Demand] simultaneously
- This system does NOT support this
```

**Priority**: CRITICAL
**Estimated Time**: 15 minutes

---

### ✅ Task 3: Clarify LSTM Validation Logic (Target vs Lag Features)
**Status**: PARTIALLY COMPLETE
**Completed**: [backend-analysis.md](2025-11-14_backend-analysis.md) (lines 374-378)

**Remaining**:
- [ ] frontend-analysis.md (clarify line 318-326 is CORRECT)
- [ ] validation-logic.md (clarify line 108-120 is CORRECT)
- [ ] variable-selection-research.md (clarify line 324 is CORRECT)
- [ ] recommendations.md (add clarification to validation rules)

**Action**: Add clarification that validation is CORRECT:
```markdown
**LSTM Target Validation (CORRECT BEHAVIOR)**:
- ✅ Frontend correctly blocks raw target variable (`Sales`) from `lstmSelectedFeatures`
- ✅ Prevents circular dependency (using Sales(t) to predict Sales(t))
- ✅ Engineered lag features (`Sales_lag_1`) ARE allowed in features
- ✅ Target history comes from Data Encoding step (creates lag features)

**Correct LSTM Feature Selection**:
- User selects "Sales" as target → blocked from feature checkboxes
- User can select "Sales_lag_1" (from encoding step) → allowed ✓
- User can select "Temperature" (external feature) → allowed ✓
```

**Priority**: CRITICAL (prevents misunderstanding of correct validation)
**Estimated Time**: 15 minutes per doc (1 hour total)

---

### ✅ Task 4: Document Feature Engineering Workflow
**Status**: COMPLETE
**Completed**: [backend-analysis.md](2025-11-14_backend-analysis.md) (lines 38-47, 675-693)

**Remaining**:
- [ ] recommendations.md (add workflow context to Option 1A)

**Action for recommendations.md**:
Update Option 1A context (around line 41):
```markdown
**Context**: In 2-step workflow:
- Data Encoding: User configures lags → creates `Sales_lag_1`, etc. → saves CSV
- Training: User selects from ALL columns (including engineered ones)
- Bug: LSTM receives `Sales_lag_1` from UI → backend re-creates lags → lag-of-lag

**Root Cause**: Training step should NOT regenerate features already in dataset
```

**Priority**: HIGH
**Estimated Time**: 15 minutes

---

## HIGH Priority Tasks (Improves Clarity)

### Task 5: Add Validation Rules Tables
**Status**: NOT STARTED
**Docs**: All except lstm_bugs_report.md

**Action**: Add table to each document:
```markdown
## Validation Rules by Algorithm

| Rule | ARIMA | XGBoost | LSTM |
|------|-------|---------|------|
| **Target** | 1 (required, radio) | 1 (required, radio) | 1 (required, radio) |
| **Features** | 0+ (auto-select) | 0+ (auto-select) | 0+ (manual only) |
| **Date** | 1 (required, radio) | 1 (required, radio) | 1 (required, radio) |
| **Empty features** | ✅ Valid (pure ARIMA) | ✅ Valid (lag-based) | ✅ Valid (univariate) |
| **Auto-selection** | ✅ Yes | ✅ Yes | ❌ No (project rule) |
| **Target in features** | ❌ Never | ❌ Never (use lags) | ❌ Never (use lag features) |
```

**Priority**: HIGH
**Estimated Time**: 10 minutes per doc (50 minutes total)

---

### Task 6: Document Auto-Selection as Project Rule
**Status**: PARTIALLY COMPLETE
**Completed**: [backend-analysis.md](2025-11-14_backend-analysis.md) (lines 49-53)

**Remaining**:
- [ ] frontend-analysis.md (add to System Constraints)
- [ ] validation-logic.md (add to validation rules)
- [ ] variable-selection-research.md (update recommendations)
- [ ] recommendations.md (add to requirements recap)

**Action**: Add explicit statement in each doc:
```markdown
**Auto-Selection Project Rule**:
- This is a PROJECT REQUIREMENT, not implementation detail
- ARIMA/XGBoost: Auto-select all numeric columns (except target/date)
- LSTM: NO auto-selection (manual checkbox selection required)
- User can always modify auto-selected features
- Minimum 0 features allowed (univariate mode valid for all)
```

**Priority**: HIGH
**Estimated Time**: 10 minutes per doc (40 minutes total)

---

## MEDIUM Priority Tasks (Nice to Have)

### Task 7: Add Cross-References
**Status**: NOT STARTED
**Doc**: [lstm_bugs_report.md](2025-11-14_lstm_bugs_report.md)

**Action**: Add after line 289:
```markdown
## Related Analysis Documents

For root cause analysis of these errors, see:
- **Error 1 (NaN)**: [Backend Analysis](2025-11-14_backend-analysis.md#feature-engineering-pipeline) (lines 166-179)
- **Errors 2-3 (validation)**: [Frontend Analysis](2025-11-14_frontend-analysis.md#bug-2-analysis) (lines 410-447)
- **All bugs**: [Recommendations](2025-11-14_recommendations.md) (proposed fixes)
- **Validation logic**: [Validation Logic](2025-11-14_validation-logic.md)
```

**Priority**: MEDIUM
**Estimated Time**: 10 minutes

---

### Task 8: Update Frontend Analysis
**Status**: NOT STARTED
**Doc**: [frontend-analysis.md](2025-11-14_frontend-analysis.md)

**Actions Needed**:
1. Add System Constraints section (after line 36)
2. Clarify LSTM validation rule is CORRECT (lines 318-326)
3. Add validation rules table
4. Add auto-selection project rule

**Priority**: MEDIUM
**Estimated Time**: 30 minutes

---

### Task 9: Update Validation Logic
**Status**: NOT STARTED
**Doc**: [validation-logic.md](2025-11-14_validation-logic.md)

**Actions Needed**:
1. Add System-Level Validation Constraints section (after line 23)
2. Clarify LSTM validation is CORRECT (lines 108-120)
3. Update validation comparison table (lines 669-682)
4. Add corrected validation rules by algorithm

**Priority**: MEDIUM
**Estimated Time**: 45 minutes

---

### Task 10: Update Variable Selection Research
**Status**: PARTIALLY COMPLETE
**Doc**: [variable-selection-research.md](2025-11-14_variable-selection-research.md)

**Completed**: Already has comprehensive research

**Actions Needed**:
1. Add terminology clarification (Task 2)
2. Clarify LSTM validation is CORRECT (line 324)
3. Update LSTM modes section (lines 145-154) with terminology
4. Update implementation issues (lines 560-584)

**Priority**: MEDIUM
**Estimated Time**: 30 minutes

---

### Task 11: Update Recommendations
**Status**: NOT STARTED
**Doc**: [recommendations.md](2025-11-14_recommendations.md)

**Actions Needed**:
1. Add "System Requirements Recap" preamble (after line 24)
2. Add workflow clarification to Option 1A (Task 4)
3. Add validation rules implementation section
4. Clarify LSTM target validation is CORRECT

**Priority**: MEDIUM
**Estimated Time**: 30 minutes

---

## LOW Priority Tasks (Polish)

### Task 12: Standardize Section Headings
**Status**: NOT STARTED
**Docs**: All

**Action**: Use consistent structure:
- ## Executive Summary
- ## System Constraints
- ## [Main Content Sections]
- ## Recommendations / Conclusions
- ## References

**Priority**: LOW
**Estimated Time**: 10 minutes per doc (1 hour total)

---

### Task 13: Add Glossary
**Status**: NOT STARTED
**Location**: Could be a separate file or added to each doc

**Glossary Terms**:
- **Univariate time series**: Single target variable
- **Multivariate inputs**: Multiple feature variables (NOT multiple targets)
- **Feature engineering**: Creating lag/rolling features
- **Auto-selection**: Automatic feature selection behavior
- **Data Encoding**: Preprocessing step before training
- **Engineered features**: Lag/rolling features created from originals
- **Raw target**: Original target column (blocked from LSTM features)
- **Target history**: Lag features of target (allowed in LSTM features)

**Priority**: LOW
**Estimated Time**: 30 minutes

---

## Summary Statistics

**Total Tasks**: 13
**Completed**: 3 (Tasks 2, 3 partial, 4)
**In Progress**: 3 (Tasks 1, 3, 8)
**Not Started**: 7

**Total Estimated Time**: ~6-7 hours for all remaining tasks

**Critical Path** (must do first):
1. Task 1: Complete System Constraints (1.3 hours)
2. Task 2: Terminology clarification (15 min)
3. Task 3: LSTM validation clarification (1 hour)
4. Task 4: Workflow documentation (15 min)

**Total Critical Path Time**: ~2.5 hours

---

## Progress Tracking

**Session 1 (2025-11-17)**:
- ✅ Added System Constraints to backend-analysis.md
- ✅ Clarified multivariate terminology in backend-analysis.md
- ✅ Updated LSTM validation logic in backend-analysis.md
- ✅ Documented feature engineering workflow in backend-analysis.md
- ✅ Created this refinement tasks document

**Remaining for Next Session**:
- [ ] Complete Tasks 1-6 (critical and high priority)
- [ ] Update frontend-analysis.md, validation-logic.md, variable-selection-research.md
- [ ] Add validation tables to all documents

---

**Document Status**: 📝 In Progress
**Last Updated**: 2025-11-17
**Next Review**: After completing critical path tasks
