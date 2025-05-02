# Assignment 2 - Machine Learning in Computational Biology (MLCB25)

This repository contains the solution to Assignment 2 of the MLCB25 course. The task was to build a complete, object-oriented machine learning pipeline to classify malignant and benign breast tumors using a dataset of 30 features derived from medical imaging.

We applied repeated nested cross-validation (rnCV) to evaluate multiple classification algorithms (Logistic Regression, Naive Bayes, LDA, SVM, Random Forest, LightGBM), followed by training a final LR model using optimal hyperparameters.

## Repository Structure

Assignment-2/
├── data/                  # Original and cleaned versions of the dataset  
│   ├── breast_cancer.csv  
│   └── breast_cancer_clean.csv  
│  
├── notebooks/             # Jupyter notebooks for EDA and model training  
│   ├── EDA.ipynb  
│   └── model_training.ipynb  
│  
├── src/                   # Python source files for reusable components  
│   └── src.py             # Includes RNestedCV and related class implementations  
│  
├── models/                # Final trained model ready for deployment  
│   └── final_model.pkl  
│  
└── README.md              # This file  

## Main Tasks Implemented

- Exploratory Data Analysis (EDA)  
- Dimensionality reduction and correlation heatmaps  
- Object-Oriented implementation of the rnCV pipeline  
- Hyperparameter tuning using inner loop CV  
- Model comparison and selection based on multiple metrics (MCC, AUC, etc.)  
- Final model training and serialization  

## Notes

- All code was written in Python using scikit-learn, pandas, and numpy.  
- Final model saved as a .pkl file for future use on unseen test data.  
- See the notebooks for full implementation details and results.
