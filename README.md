# spikes_detection_fcbg
Four pipelines for automatic epileptic spikes detection

### Pipelines description ###

All pipelines imply simple preprocessing, such as resampling to 250 Hz and filtering from 2 to 35 Hz.

*Naive pipeline (`naive`):*
Operates only on the channel with the highest variance. Applies standardization to this channel and make classification by variance thresholding.
Yields the highest rate of false positives.

*Simple SVM (`var1_svm`):*
Operates only on the channel with the highest variance. Applies standardization to this channel.
Computes 44 features for each epoch and uses Support Vector Machines for classification.

*Simple AdaBoost (`var1_abdt`):*
Operates only on the channel with the highest variance. Applies standardization to this channel.
Computes 44 features for each epoch and uses AdaBoost for classification.
Yields the best detection of spikes within seizures, however, can lead to increased false positives.

*Full pipeline (`full_pipeline_svm`):*
Aggregates information from all 16 channel using PCA. Adds a channel obtained through applying Teager-Kaiser Nonlinear Operator and PCA. 
Computes 44 x 2 channels = 88 features for each epoch and uses Support Vector Machines for classification.
Only applicable for 16 channel EEG recordings. Tested with only one particular montage.



### Usage ###

First, make sure that all the requirements are satisfied/ there exit an appropriate conda environment.

To run program for spikes detection add configurations to `config.ini` file (can be opened with any text editor).

Then use conda terminal to run the program:

```commandline
 $ cd <path_to "spikes_detection_fcbg" folder>
 $ conda activate spikes_detection
 $ python make_predictions.py
```

### Requirements ###

Tools:
- [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html "Conda installation") (Correct woek was tested with conda == 4.83, ...)

Python libraries:
- mne == 0.22.0
- sklearn == 0.24.1
- scipy == 1.4.1
- joblib == 0.15.1
  numpy == 1.18.2
- imbalanced-learn == 0.8.0
- [antropy](https://github.com/raphaelvallat/antropy "Antropy lib") == 0.1.4
- [PyWavelets](https://github.com/PyWavelets/pywt "PyWavelets lib") == 1.1.1

[comment]: <> (- pandas == 1.0.3)
Can simply create conda environment from `environment.yml` file. 
For this open conda terminal, go to folder with .yml file and run:

```
$ conda env create -f environment.yml
```

An environment with a name `spikes_detection` will be created.
To activate the environment in future run:

```
$ conda activate spikes_detection
```