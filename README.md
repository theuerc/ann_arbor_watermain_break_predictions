# Water Mains Breaks Predictions
Final project for SI 670, Applied Machine Learning, at the University of Michigan 
Group members: Coulton Theuer, Bella Karduck, Haley Johnson

## Objective
This project tries to predict if a water main will break in the next 3 years in Ann Arbor, Michigan

## Data
Data was provided to us by the City of Ann Arbor and can be found in ```data/raw```

Key datasets include: 
* ```All_Watermains_Attributes.csv```: information about all water mains in the city's distribution system
* ```Watermain_Attributes_Soil_Ph.csv```: soil information and work order IDs for water mains that have broken in the past
* ```Watermain_Breaks_Reporting.csv```: breaks reported to the City of Ann Arbor between 2013-2023


## Directory Structure
------------

The directory structure of your new project looks like this: 

```
├── README.md          
├── data
│   ├── transformed    <- Transformed data, include training and testing sets 
│   └── raw            <- Raw data from the City of Ann Arbor

├── models             <- Model traning & development
│   └── 01_baseline    <- Baseline model for comparison 
│   └── 02_models      <- Scratch notebook for developing models
│   └── 02_models_cleaned    <- Cleaned up model, random forest 
│   └── 02_svm         <- SVM model, attempting to replicate results of Syrcause paper
│   └── top_152        <- 15 pipes that are most likely to break in the next 3 years
│
├── notebooks          <- Data mainpulation & exploration 
│   └── 01_EDA         <- Combining datasets & exploratory analysis
│   └── 02_train_Test_val_split   <- Split data based on time cutoffs
│   └── 03_svm_model   <- Ignore, saved to notebooks by accident
│
├── syrcause_public   <- Cloned repository from the Syrcause paper we used as inspiration for our project
│                        We did not use this code in our project, it was mostly for our refernece
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│─ ├── utils          <- Helper functions to process data for train/test files
│   
└──
```
