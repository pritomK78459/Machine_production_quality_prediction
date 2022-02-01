# import packages
from venv import create
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
import argparse


def create_hourly_processing_features(features: pd.DataFrame) -> pd.DataFrame:

    # preprocess features to match the hourly records of the target

    features["hour_data"] = features["date_time"].apply(lambda x: x.strftime("%Y-%m-%d-%H")) # creating unique hour features
    training_data = pd.DataFrame() # initializing the training dataframe

    # refrenced from https://www.kaggle.com/alexkaggle95/production-quality-prediction-mae-6-954

    # iterating through every hour in hour data
    for hour in tqdm(features.hour_data.unique()):
        hour_data = features.loc[features.hour_data == hour] # getting the features set of the current hour
        ah = list(hour_data["AH_data"].unique())[0] # getting AH_data since it will be unique throuhgout the hour.
        hour_data = hour_data.iloc[:,1:] # slicing the hour data and removing the date_time feature

        # reindexing the data based on hour_data and AH_data
        # stackig the 60x16 data into 960,1 dataframe  
        hour_data = pd.DataFrame(hour_data.set_index(["hour_data", "AH_data"]).stack())
        hour_data = hour_data.reset_index() # now reset the index

        # after transposing the data there are two rows level_2 and 0 
        # level_2 contains columns names which are not useful as a row for the data so we will drop that 
        hour_data = hour_data[["level_2", 0]].T.drop('level_2')
        hour_data["hour_data"] = hour # here we will assign the hour
        hour_data["AH_data"] = ah # here we will assig the AH_data

        training_data = pd.concat([training_data, hour_data]) # after the row is created we can add it the dataframe initialized at the beginning
    
    training_data = training_data.dropna()
    return training_data



def preprocess_target(target:pd.DataFrame) -> pd.DataFrame:

    # preprocessing target date_time feature
    target['train_date'] = pd.to_datetime(target['date_time']) - datetime.timedelta(minutes=5) # shifting the time 5 mins back
    target['train_date'] =  pd.to_datetime(target['train_date']) 
    target['train_date'] = target['train_date'].apply(lambda x: x.strftime("%d-%m-%Y-%H")) # reformattig the datetime

    return target

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', help='features dataset path', type=str)
    parser.add_argument('--target', help='target dataset path', type=str)
    parser.add_argument('--output', help='output folder path', type=str)
    parser.set_defaults(output='./datasets/preprocessed_datasets/')
    args = parser.parse_args()

    if args.features is not None:
        try:
            features = pd.read_csv(args.features, parse_dates=True)
        except:
            raise "The dataset does not exists in the specified path, please check the path!"
    if args.target is not None:
        try:
            target = pd.read_csv(args.target, parse_dates=True)
        except:
            raise "The dataset does not exists in the specified path, please check the path!"
    if args.output is not None:
        output_folder = args.output


    # preprocessed_features = create_hourly_processing_features(features)

    preprocessed_target = preprocess_target(target)

    # preprocessed_features.to_csv(output_folder + 'preprocessed_features.csv', index=False)
    preprocessed_target.to_csv(args.output + 'preprocessed_target.csv', index=False)  


