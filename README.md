# RealCause for Casual Decision Variants (CDVs)

This repository extends the original [RealCause framework](https://github.com/bradyneal/realcause) to support **Casual Decision Variants (CDVs)** - a novel approach for improving causal inference in business processes. The extension focuses on partitioning data based on process behavior patterns to enhance the estimation of causal effects.

## Overview

The original RealCause framework provides realistic benchmarks for causal inference methods by fitting generative models to data with assumed causal structures. This extension adds support for:

- **Process variant analysis**: Identifying and leveraging case process patterns
- **Variant-specific causal modeling**: Training separate models for different process behaviors  
- **Comparative evaluation**: Benchmarking global vs. variant-specific approaches
- **Business process mining integration**: Specialized tools for event log analysis

## Key Innovation: Casual Decision Variants (CDV)

Casual Decision Variants represent different behavioral patterns in business processes. Instead of training a single global model on all data, this approach:

1. **Identifies process patterns** based on pre-treatment activity sequences and measurements
2. **Partitions data** into homogeneous variants with similar process behaviors
3. **Trains specialized models** for each variant (with global fallback)
4. **Evaluates causal effects** using variant-specific vs. global approaches

This methodology can improve causal inference accuracy by reducing confounding from process heterogeneity.

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd realcause-for-cdvs
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the required datasets in the `datasets/` folder.

## Repository Structure

The repository contains three main analysis notebooks that demonstrate the complete CDV methodology:

### 01_sepsis_analysis_realcause_preparation.ipynb
**Data Preparation and Feature Engineering**

This notebook transforms raw business process event logs into RealCause-compatible format:

- **Data Loading**: Imports Sepsis event log data from XES format
- **Admission Decision Extraction**: Identifies treatment decisions (IC vs. NC admission)
- **Feature Engineering**: Extracts pre-admission activities, measurements, and sequences
- **Variant Analysis**: Analyzes process pattern distributions and creates variant groups
- **Data Transformation**: Converts to RealCause format with treatment (t), outcome (y), and features
- **Quality Assessment**: Validates data quality and handles outliers

**Output**: `datasets/sepsis_cases.csv` - Clean dataset ready for causal analysis

### 02_cdv_modeling.ipynb
**RealCause Model Training and Synthetic Data Generation**

This notebook implements the core CDV methodology:

- **Model Training**: Trains RealCause generative model on Sepsis event log data
- **Architecture Comparison**: Evaluates different neural network architectures
- **Synthetic Data Generation**: Creates training data with counterfactual outcomes
- **Variant Assignment**: Partitions data based on feature patterns (top-k 'elbow' approach)
- **Data Preparation**: Creates variant-specific datasets for specialized modeling
- **Multi-Seed Experiments**: Runs comprehensive causal inference experiments comparing:
  - **Global Method**: Single model trained on all data
  - **CDV Method**: Variant-specific models with global fallback
- **Estimator Evaluation**: Tests multiple causal estimators (S-Learner, T-Learner, X-Learner, DR-Learner, Double-ML)

**Output**: Model checkpoints and experimental results for causal analysis

### 03_causal_analysis.ipynb
**Results Analysis and Statistical Evaluation**

This notebook provides analysis of the CDV approach effectiveness:

- **ATE Analysis**: Average Treatment Effect estimation and bias-variance decomposition
- **CATE Analysis**: Conditional Average Treatment Effect evaluation
- **Statistical Testing**: Significance tests comparing global vs. CDV methods
- **Bias-Variance Decomposition**: Analyzes sources of prediction improvement
- **Effect Size Calculation**: Quantifies practical significance using Cohen's d
- **Visualization**: Creates plots and statistical summary tables


**Key Metrics Evaluated**:
- Mean Squared Error (MSE) for treatment effects
- Bias-variance decomposition
- Statistical significance testing
- Effect sizes and confidence intervals

## CDV Utils Package

The `cdv_utils/` package provides specialized utilities:

- `sepsis_loading.py`: Event log data processing
- `feature_extraction.py`: Business process feature engineering  
- `causal_modeling.py`: Variant assignment and data partitioning
- `experiment_runner.py`: Multi-seed experimental framework
- `results_analysis.py`: Statistical analysis and testing
- `visualization.py`: Plotting and visualization tools

## Citation

If you use this CDV extension in your research, please cite both the original RealCause paper and this extension:

```bibtex
@article{,
  title={},
  author={[]},
  year={},
  note={Extension of RealCause framework}
}
```

## License
This project extends the original RealCause framework and follows the same licensing terms.