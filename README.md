# Deep Kernel Learning
This repository contains the source code for deep kernel learning described in the paper: "Are you sure it’s an artifact? Artifact detection and uncertainty quantification in histological images"

link to the paper will be added soon.

# Requirements
- Python >= 3.6.7
- Numpy == 1.23.4
- Pytroch == 2.2.0
- Gpytorch == 1.9.0
- Pandas == 1.5.1
- deepsig == 1.2.6
- opencv-python == 4.7.0.68
- pyvips == 2.2.1
- openslide-python == 1.2.0
- Matploblib
- Scipy
- Scikit-learn
- Seaborn

# Abstract
Will be updated soon.

# Results
Will be added after publishing

# How to use the code
Please install requirements.txt or Python dependencies separately.
Update paths to the processed dataset and path to save experiment results.
## Dataset 

```
- path_to\blur_dataset
      - training
           -- artifact_free
            -- blur
      - validation
            -- artifact_free
            -- blur
       - test
            -- artifact_free
            -- blur
```

- Train models using train_dcnn.py for SOTA DCNNs mentioned in the paper
- Train DKL models using train_dkl.py, choosing specific architectures and hyperparameters.
- Train Baseline models using train_baseline.py

- Use paths to experiment directories for best_weights.dat and run inference.py for test set, TCGAFocus and FocusPath
- Use predicted excel sheets to create confidence plots using plot_confidence.py
  
# Publically available datasets
- FocusPath: https://zenodo.org/records/3926181
- TCGAFocus: https://zenodo.org/records/3910757
  
Use transform_tcga.py to transform the dataset for running inference models.

## Private Dataset      
EMC dataset mentioned in the paper will soon be released and link will be added here.
   
# How to cite our Work
The code is released free of charge as open-source software under the GPL-3.0 license. Bibtex citations will be made available soon.

