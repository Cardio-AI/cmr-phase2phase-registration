Deep Learning-based aligned strain from cine MRI improves the detection of fibrotic myocardial tissue in patients with Duchenne Muscular Dystrophy
==============================
This repository provides a pipeline to derive cardiac-phase-specific strain values from one key frame to the next and throughout the cardiac cycle. The image source are short stacks of short axis cine SSFP cardiac MR images.

**Please Note:**
Repository is work in progress and contains the source code for many different experiments, as the corresponding paper is under review. Once the manuscript and method is accepted for publication further descriptions and instructions on how to reproduce the results of the accepted manuscript are provided.

Abstract:
--------
Background: Rare pathologies like Duchenne muscular dystrophy (DMD) are monitored using
late gadolinium-enhanced (LGE) cine Cardiac Magnetic Resonance (CMR) sequences to track
myocardial fibrosis. However, reducing gadolinium exposure is crucial due to its invasive and
time-consuming nature. Strain analysis from cine (non-contrast) CMR serves as a valuable
indicator for detecting abnormal cardiac function.
Purpose: Unfortunately, traditional strain lack temporal alignment between patients. Peak or
end-systolic strain disregard diastolic deformation patterns, while markers like early diastolic
strain-rates necessitate manual frame selection.
Materials and Methods: Our Deep Learning pipeline detects five key frames throughout the
cardiac cycle, allowing for temporally aligned, phase-specific strain analysis across patients by
deriving them from one key frame to the next. We evaluated the effectiveness of these strain
values in identifying abnormal deformations associated with fibrotic segments in a
retrospective study of 57 patients and assessed reproducibility in 82 patients, comparing our
method with existing feature-tracking and DL-based strain methods. The study involved cine
CMR from 139 DMD patients collected in one centre between 2018 and 2023.
Results: Aligned strain revealed five times more differences (29 vs. 5, p < 0.01) between
fibrotic and non-fibrotic segments and identified abnormal diastolic deformation patterns often
missed by traditional strain-methods. Additionally, it enhanced model robustness for fibrosis
detection, improving specificity by 40%, accuracy by 17%, and accuracy for detecting fibrosis
in DMD with preserved ejection fraction by 61%.
Conclusion: The pathology-independent technique enables motion-based detection of
myocardial dysfunction, potentially reducing contrast agent exposure, facilitating detailed interpatient
strain analysis, and allowing precise tracking of disease progression in DMD.

Checkout commit:
- [commit](https://github.com/Cardio-AI/cmr-phase2phase-registration/commit/15a15c8bd9abf2319027357397597b4931db3765)
- date: 03.11.2022
- comit message: "added slurm jobid int exp file names"
- SHA: 15a15c8bd9abf2319027357397597b4931db3765

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like 'make environment' or 'make requirement'
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── metadata       <- Excel and csv files with additional metadata
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── predicted      <- Model predictions, will be used for the evaluations
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. 
    │   ├── Dataset        <- call the dataset helper functions, analyze the datasets
    │   ├── Evaluate       <- Evaluate the model performance, create plots
    │   ├── Predict        <- Use the models on new data
    │   ├── Train          <- Train a new model
    │   └── Test_IO        <- IO tests
    │   └── Test_Models    <- Tensorflow functional or subclassing tests
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── configs        <- Experiment config files as json
    │   ├── figures        <- Generated graphics and figures to be used in reporting
    │   ├── history        <- Tensorboard trainings history files
    │   └── tensorboard_logs  <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- Makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Helper functions that will be used by the notebooks.
        ├── data           <- create, preprocess and extract the nrrd files
        ├── models         <- Modelzoo, Modelutils and Tensorflow layers
        ├── utils          <- Metrics, callbacks, io-utils, notebook imports
        └── visualization  <- Plots for the data, generator or evaluations
Paper:
--------
- link to paper if accepted

- info for cite

Setup native with OSX or Ubuntu
------------
### Preconditions: 
- Python 3.6 locally installed 
(e.g.:  <a target="_blank" href="https://www.anaconda.com/download/#macos">Anaconda</a>)
- Installed nvidia drivers, cuda and cudnn 
(e.g.:  <a target="_blank" href="https://www.tensorflow.org/install/gpu">Tensorflow</a>)

### Local setup
### Create a new project from the template:
------------

0. Clone repository
```
git clone %repo-name%
cd %repo-name%
```
1. Create a new environment or from an environment.yaml file
```
conda create --name %PROJEKTNAME% --python=3.8
or
conda env create --file environment.yaml
```

2. Activate environment
```
conda activate ax2sax
```
3. Install a helper to automatically change the working directory to the project root directory
```
pip install --extra-index-url https://test.pypi.org/simple/ ProjectRoot
```
4. Create a jupyter kernel from the activated environment, this kernel will be visible in the jupyter lab started from the base environment
```
pip install ipykernel
python -m ipykernel install --user --name ax2sax --display-name "ax2sax kernel"
```
5. Deactivate the current environment and run Jupyter lab
```
conda deactivate
jupyter lab
```

### Enable interactive widgets in Jupyterlab

Pre-condition: nodejs installed globally or into the conda environment. e.g.:
```
conda install -c conda-forge nodejs
```
Install the jupyterlab-manager which enables the use of interactive widgets
```
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

Further infos on how to enable the jupyterlab-extensions:
[JupyterLab](https://ipywidgets.readthedocs.io/en/latest/user_install.html#installing-the-jupyterlab-extension)
