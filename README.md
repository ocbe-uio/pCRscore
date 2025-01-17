# pCRscore

Python package for predicting pathological Complete Response (pCR) scores according to [Azimzade et al. (2023)](https://www.biorxiv.org/content/10.1101/2023.09.07.556655). The original source code is available at https://github.com/YounessAzimzade/XML-TME-NAC-BC.

# Installation

This package is under development and not yet available on PyPI. You can, nonetheless, install the development version directly from GitHub by running this command on your terminal (requires git):

```bash
pip install git+https://github.com/ocbe-uio/pCRscore.git
```

# Expected Input

This package takes cell fractions, likely from spatial proteomics, scRNA seq or estimations using deconvolution methods and clinical out come (pCR vs RD- residual disease). It is possible to provide the "Cohort" info   alongside these information, otherwise the code randomly assigns the rows to discovery and validation cohorts. Data should have a structure as below:

| Sample        | CellType 1       | CellType 2 | ...       | Response        | Cohort |  
|-----------------|-----------------|----------------|-----------------|-----------------|----------------|
| TX1 | 15 | 4| ... | pCR | Discovery|
|  TX2 | 0| 12| ... | RD | Validation|
| TX3| 5| 17| ... | RD | Validation|


# Expected Output
For each cell type, a pCR score is assined and provided in csv file. Ucertanity values for these scores are also provided. 

# References

Explainable Machine Learning Reveals the Role of the Breast Tumor Microenvironment in Neoadjuvant Chemotherapy Outcome
Youness Azimzade, Mads Haugland Haugen, Xavier Tekpli, Chloé B. Steen, Thomas Fleischer, David Kilburn, Hongli Ma, Eivind Valen Egeland, Gordon Mills, Olav Engebraaten, Vessela N. Kristensen, Arnoldo Frigessi, Alvaro Köhn-Luque
bioRxiv 2023.09.07.556655; doi: https://doi.org/10.1101/2023.09.07.556655
