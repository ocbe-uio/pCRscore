# pCRscore

Python package for predicting pathological Complete Response (pCR) scores using explainable machine learning to analyze the role of the breast tumor microenvironment in neoadjuvant chemotherapy outcomes. This package implements the methodology described in [Azimzade et al. (2023)](https://www.biorxiv.org/content/10.1101/2023.09.07.556655). The original source code is available at https://github.com/YounessAzimzade/XML-TME-NAC-BC.

## Overview

Neoadjuvant Chemotherapy (NAC) is the established treatment approach for patients with large breast tumors, involving the administration of chemotherapy drugs prior to surgical removal of the tumor. Pathological Complete Response (pCR), which refers to the complete elimination of cancer cells in the breast and auxiliary lymph nodes following NAC, serves as a highly favorable prognostic biomarker.

Breast tumors are complex ecosystems comprising cancer cells, normal epithelial cells, immune cells, and stromal cells. This package leverages explainable machine learning (XML) to robustly explore the associations between different cell phenotype fractions and the response to NAC in the general population as well as different subtypes of breast tumors.

The package implements a novel pipeline that:
- Uses Support Vector Machine (SVM) classifiers to predict pCR from cell type fractions
- Employs SHAP (SHapley Additive exPlanations) values for model interpretability
- Calculates **pCR Score** as a robust metric for the association of cell type fractions with the probability of achieving pCR
- Validates findings across discovery and validation cohorts
- Provides uncertainty estimates through confidence intervals

**Note**: While this package was developed and validated on breast cancer data, it is a generic framework that can be applied to different cancer types, as long as the input data follows the structure described in the "Expected Input" section below. The methodology is agnostic to the specific cell types or cancer type, making it adaptable to various tumor microenvironments and treatment response scenarios.

## Key Findings from the Paper

The analysis of more than 2000 breast tumor samples revealed that multiple cell types exhibit distinct associations with pCR within different tumor subtypes. Notably:

- **Dendritic cells (DCs)** exhibit a negative association with pCR in Estrogen Receptor positive (ER+, Luminal A/B) tumors, while showing a positive association with pCR in ER- (Basal-like/HER2-enriched) tumors
- Analysis of spatial cyclic immunofluorescence data and imaging mass cytometry data showed significant differences in the spatial distribution of DCs between ER subtypes
- The findings on 28 different cell types provide a comprehensive understanding of the role played by cellular components of the Tumor Microenvironment (TME) in NAC outcomes

## Installation

This package is under development and not yet available on PyPI. You can, nonetheless, install the development version directly from GitHub by running this command on your terminal (requires git):

```bash
pip install git+https://github.com/ocbe-uio/pCRscore.git
```

## Expected Input

This package takes cell fractions, likely from spatial proteomics, single-cell RNA sequencing (scRNA-seq), or estimations using deconvolution methods (such as CIBERSORTx), along with clinical outcome data (pCR vs RD - residual disease). 

The data should include:
- **Cell type fractions**: Numerical columns representing the fraction of each cell type in the tumor sample
- **Response**: Binary outcome variable indicating 'pCR' (pathological complete response) or 'RD' (residual disease)
- **Cohort** (optional): Column indicating whether each sample belongs to 'Discovery' or 'Validation' cohort. If not provided, the code randomly assigns rows to discovery and validation cohorts with a 50/50 split.

Data should have a structure as below:

| Sample | CellType 1 | CellType 2 | ... | Response | Cohort |
|--------|------------|------------|-----|----------|--------|
| TX1    | 15         | 4          | ... | pCR      | Discovery |
| TX2    | 0          | 12         | ... | RD       | Validation |
| TX3    | 5          | 17         | ... | RD       | Validation |

### Supported Cell Types

The package can analyze various cell types commonly found in breast tumor microenvironments, including but not limited to:

- **Immune cells**: B cells (Memory, Naive), T cells (CD4+, CD8+), NK cells, NKT cells, Dendritic cells (DCs), Macrophages, Monocytes, Plasmablasts
- **Stromal cells**: Cancer-associated fibroblasts (CAFs) - MSC/iCAF-like, myCAF-like, Perivascular-like cells (PVLs) - Differentiated, Immature
- **Endothelial cells**: ACKR1, CXCL12, LYVE1, RGS5
- **Epithelial cells**: Luminal Progenitors, Mature Luminal, Myoepithelial, Normal Epithelial, Cancer Cells

## Methodology

The pipeline follows these key steps:

1. **Data Preprocessing**: 
   - Normalizes cell fractions by the 99th percentile
   - Removes outliers (values > 1 after normalization)
   - Handles binary and categorical variables automatically

2. **Model Training**:
   - Trains an SVM classifier with RBF kernel on the discovery cohort
   - Uses class-balanced weights to handle imbalanced datasets
   - Standardizes features for optimal SVM performance

3. **SHAP Analysis**:
   - Computes SHAP values for each sample and cell type using KernelExplainer
   - SHAP values quantify the contribution of each cell type fraction to the prediction

4. **pCR Score Calculation**:
   - Combines normalized cell fractions with their corresponding SHAP values
   - Fits linear models (SHAP ~ Fraction) for each cell type
   - The coefficient from this linear model represents the association strength
   - Validates associations by requiring consistent direction in both discovery and validation cohorts
   - Calculates pCR Score as a normalized metric (clipped to [-1, 1]) where:
     - **Positive scores** indicate cell types associated with higher pCR probability
     - **Negative scores** indicate cell types associated with lower pCR probability (residual disease)

5. **Uncertainty Estimation**:
   - Provides confidence intervals (99% CI) for each pCR Score
   - Only cell types with consistent associations (same direction) in both cohorts and significant confidence intervals are retained

## Expected Output

For each cell type, a pCR score is assigned and provided. The output includes:

- **pCR Score**: Normalized association metric ranging from -1 to 1
- **Confidence Intervals**: Lower and upper bounds (LI, HI) for uncertainty quantification
- **Coefficient**: Raw linear model coefficient from SHAP vs Fraction regression

The results can be visualized as a bar plot showing pCR scores for each cell type, with error bars representing confidence intervals.

![image](https://github.com/user-attachments/assets/d76898ee-5e31-40fe-9c91-941080735fb4)

## Usage Example

```python
import pandas as pd
from pCRscore import svm, pipeline

# Load your data
data = pd.read_csv('your_data.csv')

# Preprocess data (splits into discovery and validation cohorts)
data_disc, data_valid = svm.preprocess(data, split_var='Cohort')

# Extract features and target
X_disc, y_disc = svm.extract_features(data_disc)
X_valid, y_valid = svm.extract_features(data_valid)

# Perform SHAP analysis on discovery cohort
shap_values_disc = svm.shap_analysis(X_disc, y_disc, nsamples=100, pandas_out=True)

# Normalize cell fraction data
cell_fractions_disc = pipeline.drop_non_float(data_disc)
cell_fractions_disc_norm = pipeline.normalize_data(cell_fractions_disc.copy())

# Combine fractions with SHAP values
combined_disc = pipeline.combine_fractions_shap(cell_fractions_disc_norm, shap_values_disc)

# Fit linear models to get pCR scores
fit_disc = pipeline.fit_line(combined_disc)

# Repeat for validation cohort and combine results
# ... (similar steps for validation cohort)

# Final pCR scores with confidence intervals
final_scores = pipeline.fit_line(combined_all, split_ci=True)

# Visualize results
pipeline.plot_fit(final_scores)
```

## Package Structure

- `pipeline.py`: Core pipeline functions for data normalization, SHAP-fraction combination, linear model fitting, and visualization
- `svm.py`: SVM model training, SHAP analysis, and preprocessing utilities
- `misc.py`: Helper functions for binary encoding and categorical variable handling

## References

**Explainable Machine Learning Reveals the Role of the Breast Tumor Microenvironment in Neoadjuvant Chemotherapy Outcome**

Youness Azimzade, Mads Haugland Haugen, Xavier Tekpli, Chloé B. Steen, Thomas Fleischer, David Kilburn, Hongli Ma, Eivind Valen Egeland, Gordon Mills, Olav Engebraaten, Vessela N. Kristensen, Arnoldo Frigessi, Alvaro Köhn-Luque

bioRxiv 2023.09.07.556655; doi: https://doi.org/10.1101/2023.09.07.556655

## License

This package is licensed under the GNU General Public License (GPL). See the LICENSE file for details.
