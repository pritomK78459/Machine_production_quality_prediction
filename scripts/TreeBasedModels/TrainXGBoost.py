# import packages

import argparse
import pickle
from random import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def train_decision_tree_regressor(X_train:pd.DataFrame, y_train:pd.Series):

    # this function will train a Xtreme Gradient Boosting Regressor on the train features and target

    XBR = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8) # initialize the regressor

    # regressor = RandomizedSearchCV(estimator=RandomForest,param_distributions=parameters, 
    #                                 n_iter=100, cv=3, verbose=2, n_jobs=-1, random_state=42)   # initialize the randomized search regressor
    
    results = XBR.fit(X_train, y_train) # train the regressor

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train decision tree model.')
    parser.add_argument('--features', help=' path to features dataset', type=str)
    parser.add_argument('--target', help='path to target dataset', type=str)
    parser.add_argument('--output', help='destination folder to save the model', type=str)
    parser.add_argument('--save', dest='save', action='store_true', help='save model')
    parser.add_argument('--no-save', dest='save', action='store_false', help='do not save the model')
    parser.set_defaults(features='datasets/preprocessed_datasets/preprocessed_features.csv',
                        target='datasets/preprocessed_datasets/preprocessed_target.csv',
                        output='Models/XGBoostBest.sav',
                        save=True)
    args = parser.parse_args()

    # prepared datasets
    features = pd.read_csv(args.features)
    features = features.dropna()
    target = pd.read_csv(args.target)

    model_output_folder = args.output

    final_df = features.join(target, how='inner') # combine feature and target datasets

    features = final_df.drop(['quality','hour_data', 'date_time','train_date'], axis=1)

    # scaling the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    target = final_df['quality']

    X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.25, random_state=42)   # splitting the datasets in to train and test sets

    results = train_decision_tree_regressor(X_train, y_train)

    # print(results.best_estimator_)  # best parameters

    predictions = results.predict(X_test)   # model predictions

    MAE_score = mean_absolute_error(y_test, predictions)    # evalutate the model on test data using mean absolute error
    MSE_score = mean_squared_error(y_test, predictions)     # evalutate the model on test data using mean squared error
    RMSE_score = np.sqrt(MSE_score)                         # evalutate the model on test data using root mean squared error

    print(f'Mean absolute error for test data: {MAE_score}')
    print(f'Mean squared error for test data: {MSE_score}')
    print(f'Root mean squared error for test data: {RMSE_score}')

    # save the model
    if args.save:
        pickle.dump(results, open(args.output, 'wb'))
