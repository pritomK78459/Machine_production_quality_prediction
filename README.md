
# Machine Production Quality Prediction

The goal is to create a model that determines the quality of items produced on a roasting process based on data arriving every minute.

The roasting machine is made up of five compartments of identical size, each with three temperature sensors. You've also gathered information on the raw material layer's height and moisture content for this task. When raw materials enter the machine, their layer height and humidity are measured. In an hour, raw materials flow through the kiln.

# Dataset

data_X.csv: It contains the records of parameters of the roasting machine which was taken every minute in the roasting process.

data_Y.csv: It contains the measurement of the quality of the product produced by the machine.   
## Problems Faced

- There is a mismatch in features recording interval and target recording interval i.e the recording of the machine parameters were taken every minute whereas the quality measurement was taken every hour.
- The dataset is big, but the features from the minutes interval also contribute to overfitting.

## Installation

Create a conda environment 

```bash
  conda create -n machine_quality_prediction python=3.8
```
    
Install requirements

```bash
    pip install -r requirements.txt
```

## Create Dataset

Run PrepareDataset.py script in the scripts directory to prepare preprocessed datasets 

```bash
  python .\scripts\PrepareDataset.py
```




## Train models

The models will be saved in the Models folder

The following models are available to train on:

* Decision Tree
* Random Forests
* Gradient Boosting
* Extreme Gradient Boosting (XGBoost)

Example:

```bash
usage: TrainDecisionTree.py [-h] [--features FEATURES] [--target TARGET] [--output OUTPUT]
                            [--save] [--no-save]

Train decision tree model.

optional arguments:
  -h, --help           show this help message and exit
  --features FEATURES  path to features dataset
  --target TARGET      path to target dataset
  --output OUTPUT      destination folder to save the model
  --save               save model
  --no-save            do not save the model
```
Example:

```bash
  python .\scripts\TreeBasedModels\TrainDecisionTree.py
```




## Using the trained models

You will require the pickle package

```python
import pickle
```

load the model

Example:

```python
model = pickle.load(open('Models/XGBoostBest.sav', 'rb'))
```

## Authors

- [@pritomK78459](https://github.com/pritomK78459)

