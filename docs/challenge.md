# Challenge dev process: 

## Created functions

This directory contains all the functions used to add features and
other actions to prepare the dataframe.


## Model

DelayModel is the class that contains the model and methods to prpeare
the training and predictions.

Also transform the dataset to a numeric matrix (dataframe) with all
features needed.

## Api 

On the API, given that is not defined exactly the features that comes
from the client, I set some of these randomly, another under
conditions, like months 1-12, etc.


In case of error the post returns error 400.

In case data is correct returns json with predictions key, a list of values


## Train the model

Prepared script to train the model with the dataset.

Once the API is running it loads the trained model.
