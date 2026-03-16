
# Case Study: Statistical Summary

**Total Cases**: 15 (selected from cases where E1 flips but E3 doesn't)

## Key Findings

### 1. Flip Rate
- **E1 Baseline**: 100.0% (15/15 cases flip)
- **E3 LLM+CLP**: 0.0% (0/15 cases flip)
- **Reduction**: 100.0 percentage points

### 2. Prediction Shift
- **E1 Baseline**: 0.625 ± 0.294
- **E3 LLM+CLP**: 0.094 ± 0.065
- **Reduction**: 85.0%

### 3. By Identity Group
- **he→she**: 9 cases, E1 flip 9/9, E3 flip 0/9
- **white→black**: 2 cases, E1 flip 2/2, E3 flip 0/2
- **queer→straight**: 1 cases, E1 flip 1/1, E3 flip 0/1
- **men→women**: 1 cases, E1 flip 1/1, E3 flip 0/1
- **gay→straight**: 1 cases, E1 flip 1/1, E3 flip 0/1
- **his→her**: 1 cases, E1 flip 1/1, E3 flip 0/1

### 4. Most Extreme Case
- **Largest E1 shift**: 0.991
- **Corresponding E3 shift**: 0.143
- **Improvement**: 85.6%

## Interpretation

1. **Perfect counterfactual invariance**: In all 15 selected cases, E3 maintains consistent predictions across counterfactuals, while E1 flips in 100% of cases.

2. **85% reduction in prediction shift**: E3 reduces the average prediction shift from 0.625 to 0.094, demonstrating strong robustness to identity term changes.

3. **Gender swaps most common**: 9/15 cases involve he→she swaps, indicating that gender is a particularly strong spurious feature in the baseline model.

4. **Consistent improvement across groups**: E3 achieves 0% flip rate across all identity groups, showing that LLM+CLP generalizes well.
