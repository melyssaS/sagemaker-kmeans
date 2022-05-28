!pip install pymongo

from __future__ import print_function

import argparse
import os
import pandas as pd

from sklearn import tree
from sklearn.externals import joblib
from pymongo import MongoClient


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Sagemaker specific arguments. Defaults are set in the environment variables.

    #Saves Checkpoints and graphs
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    #Save model artifacts
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    #Train data
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    #file = os.path.join(args.train, "person.csv")
    #dataset = pd.read_csv(file, engine="python")
    #dataset.head()
    #dataset.columns
    
    client = MongoClient('mongodb+srv://findme-user:fubdne-passwrod@findme-db.gd6yh.mongodb.net/find-me?retryWrites=true&w=majority')
    collection = client["find-me"]["kmeans"]
    data = collection.find()
    data = list (data)
    
    #Clean Data
    df = pd.DataFrame (data)
    df = dataset.dropna()
    
 
    df["birthDate_range"] = df[ "birthDate" ].apply(range_from_value)

    # Encoding categorical data
    from sklearn.preprocessing import OrdinalEncoder
    columns_of_interest = [ 'eyeColor', 'skinColor', 'hairType','height',"documentType","birthDate_range","sex"]
    ord_enc = OrdinalEncoder()
    df[columns_of_interest] =  ord_enc.fit_transform(df[columns_of_interest])
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    main_cols =[ 'eyeColor', 'skinColor', 'hairType','height',"documentType","birthDate_range","sex"]
    scale = scaler.fit_transform(df[main_cols ])
    df_scale = pd.DataFrame(scale, columns = main_cols );
    df_scale.head()
    X = df_scale[main_cols].values
    df = df_scale

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters = 3, init = 'k-means++',max_iter = 300 ,n_init =  10,random_state = 0)
    y_kmeans = kmeans.fit(X)
    df['KMeans_Clusters'] = kmeans.labels_

    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(kmeans, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    """Deserialized and return fitted model
    
    Note that this should have the same name as the serialized model in the main method
    """
    kmeans = joblib.load(os.path.join(model_dir, "model.joblib"))
    return kmeans

def range_from_value(value):
    year = value[:-8]
    return '%s0 - %s9'%(year[:-1],year[:-1])
    
