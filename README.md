# Machine Learning Assignments - Nearest Neighbour, Conformal Prediction and SVMs

## Overview
These assignments were completed as part of the coursework for a 3rd year machine learning module.
It involved implementing and evaluating different machine learning algorithms such as K-Nearest Neighbour (KNN), Conformal Prediction and Support Vector Machines (SVMs).
The assignments use various different datasets to demonstrate these techniques.

## Requirements
- jupyter - https://jupyter.org
- numpy - https://numpy.org
- matplotlib - https://matplotlib.org
- scikit-learn - https://scikit-learn.org/stable/

Jupyter can be installed using the following command:
```bash
pip install jupyter
```

All the remaining libraries can be installed using pip by running the following command:
```bash
pip install numpy matplotlib scikit-learn
```

## Build Instructions
1. Clone the repository:
```bash
git clone https://github.com/olijbrown/MLExamples
```

2. Open Jupyter notebook:
```bash
jupyter notebook
```

3. Open the cloned repository and run

## Datasets
- Iris Dataset: Contains 150 samples of Iris flowers with 3 species and 4 features. Used for multi-class classification tasks to predict the species of Iris based on the features.
- Ionosphere Dataset: Consists of 351 radar data samples. Used for binary classification by determining if the returns by are radar are either 'good' or 'bad'.
- Wine Dataset: Dataset of wines cultivated in 3 regions of italy. Used for multi-class classification tasks to distinguish between the 3 different cultivators of wine.
- Zip Dataset: Dataset consisting of various handwritten digits (0-9). Used for multi-class classification tasks to classify the images into one of the 10 digit classes.

## Project Components

### Data Preparation
- Loading Datasets: Project loads datasets using 'sklearn.datasets' and 'numpy.genfromtxt'.
- Data Splitting: Datasets are split into the training and test sets using 'train_test_split'.

### K-Nearest Neighbour
- Implementation: 1-Nearest Neighbour and K-Nearest Neighbour are implemented to predict the class of a sample by finding the nearest neighbour(s) in the training set.
- Error Calculation: The error rate of the K-Nearest Neighbour is calculated and displayed.
- Visualisation: The error rates of the K-Nearest Neighbour are plotted against different K values.

### Conformal Prediction
- Implementation: A conformal prediction function is implemented to provide p-values for test samples based on conformity measures.
- Error Calculation: The average false p-values are calculated for the test set. 

### Support Vector Machines
- Cross-Validation: The cross_val_score function is used to evaluate the SVM model with cross-validation.
- Hyperparameter Tuning: The GridSearchCV function is used to find the best parameters for the SVM model using a pipeline with different scalers.
- Error Calculation: The error rates of the SVM models are calculated on the test sets.
