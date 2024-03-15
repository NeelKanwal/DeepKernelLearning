# Deep Kernel Learning
This repository contains the source code for deep kernel learning described in the paper: "Are you sure it’s an artifact? Artifact detection and uncertainty quantification in histological images", published in Computerized Medical Imaging and Graphics Journal.

[link to the paper: ](https://www.sciencedirect.com/science/article/pii/S0895611123001398)

<img width="1477" alt="image" src="https://github.com/NeelKanwal/DeepKernelLearning/assets/52494244/91cf8f5f-afd9-4b2c-9e41-cf4e9b32b868">


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
Modern cancer diagnostics involves extracting tissue specimens from suspicious areas and conducting histotechnical procedures to prepare a digitized glass slide, called Whole Slide Image (WSI), for further examination. These procedures frequently introduce different types of artifacts in the obtained WSI, and histological artifacts might influence Computational Pathology (CPATH) systems further down to a diagnostic pipeline if not excluded or handled. Deep Convolutional Neural Networks (DCNNs) have achieved promising results for the detection of some WSI artifacts; however, they do not incorporate uncertainty in their predictions. This paper proposes an uncertaintyaware Deep Kernel Learning (DKL) model to detect blurry areas and folded tissues, two types of artifacts that can appear in WSIs. The proposed probabilistic model combines a CNN feature extractor and a sparse Gaussian Processes (GPs) classifier, which improves the performance of current state-of-the-art artifact detection DCNNs and provides uncertainty estimates. We achieved 0.996 and 0.938 F1 scores for blur and folded tissue detection on unseen data, respectively. In extensive experiments, we validated the DKL model on unseen data from external independent cohorts with different staining and tissue types, where it outperformed DCNNs. Interestingly, the DKL model is more confident in the correct predictions and less in the wrong ones. The proposed DKL model can be integrated into the preprocessing pipeline of CPATH systems to provide reliable predictions and possibly serve as a quality control tool.

<img width="1498" alt="image" src="https://github.com/NeelKanwal/DeepKernelLearning/assets/52494244/322f2232-11bc-4f77-bf4f-731050a0b4ab">


# Results
<img width="1471" alt="image" src="https://github.com/NeelKanwal/DeepKernelLearning/assets/52494244/a5548c55-a358-4cac-88ef-111dd24be829">

<img width="1493" alt="image" src="https://github.com/NeelKanwal/DeepKernelLearning/assets/52494244/cb727c29-ee67-4967-91db-fc230c838aae">

<img width="749" alt="image" src="https://github.com/NeelKanwal/DeepKernelLearning/assets/52494244/395cb5b9-088a-4eb4-960c-06650239ba2d">

<img width="800" alt="image" src="https://github.com/NeelKanwal/DeepKernelLearning/assets/52494244/b2a81d5f-d760-4e3d-a5a2-6849d20d707f">

<img width="439" alt="image" src="https://github.com/NeelKanwal/DeepKernelLearning/assets/52494244/57234cd4-5348-448b-a243-1197c7084c7e">


# How to use the code
Please install requirements.txt or Python dependencies separately.
Update paths to the processed dataset and path to save experiment results.
## Dataset 

The dataset is publicaly available at Zenodo. https://zenodo.org/records/10809442. 

You can use D40x directory and corresponding folders with artifacts to organize in the following order.

For folded tissue, D20x is used in this work for development and D40x for testing the folded tissue DKL models. 

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
  
# Other Publically available datasets (used in external validation)
- FocusPath: https://zenodo.org/records/3926181
- TCGAFocus: https://zenodo.org/records/3910757
  
Use transform_tcga.py to transform the dataset for running inference models.

## Private Dataset      
EMC dataset mentioned in the paper will soon be released and link will be added here.
   
# How to cite our Work
The code is released free of charge as open-source software under the GPL-3.0 license. Please cite this work if you our code, data or build on top of it.

```
@article{kanwal2023you,
  title={Are you sure it’s an artifact? Artifact detection and uncertainty quantification in histological images},
  author={Kanwal, Neel and L{\'o}pez-P{\'e}rez, Miguel and Kiraz, Umay and Zuiverloon, Tahlita CM and Molina, Rafael and Engan, Kjersti},
  journal={Computerized Medical Imaging and Graphics},
  pages={102321},
  year={2023},
  publisher={Elsevier}
}
```
Other works on HistoArtifact datasets:
1. Vision-Transformers-for-Small-Histological-Datasets-Learned-Through-Knowledge-Distillation: https://github.com/NeelKanwal/Vision-Transformers-for-Small-Histological-Datasets-Learned-Through-Knowledge-Distillation
2. Quantifying-the-effect-of-color-processing-on-blood-and-damaged-tissue-detection: https://github.com/NeelKanwal/Quantifying-the-effect-of-color-processing-on-blood-and-damaged-tissue-detection
3. Are you sure it’s an artifact? Artifact detection and uncertainty quantification in histological images: https://github.com/NeelKanwal/DeepKernelLearning
