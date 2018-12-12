How to run our code:
1. Run create_data.py
    -this will create 3 dictionaries, one for the graph, training set, and test set, that will be used to extract features from the dataset and are pickled into the data folder.
2. Run create_features_v2.py
    - this creates the feature array for the training set and pickles that.
3. Run create_features_v3.py
    -this creates the feature array for the test set and pickles that.
4. Run modelSelection.py
    -this runs our learning on the test set and saves the predictions in a text file in the data folder.
After this, the predictions will be in a text file named submission_best_random_forest.txt in the data folder.

cleaning.py was used to explore the dataset and analyze outliers.
create_features_v1.py was the experimental feature extraction file. We played around with different features to extract in this file.
create_features_v4 and 5.py were used to create the features for the train and test set with additional features appended to it. 