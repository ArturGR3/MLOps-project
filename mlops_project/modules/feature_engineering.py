import os
import re
import warnings
import pandas as pd
from openfe import OpenFE, transform, tree_to_formula
import contextlib
from autogluon.tabular import TabularDataset, TabularPredictor
import numpy as np
from sklearn.model_selection import train_test_split


class FeatureEnginering:
    def __init__(self, competition_name, target_column):
        """
        Initialize the FeatureEngineering class with competition name and target column.

        Parameters:
        competition_name (str): The name of the competition.
        target_column (str): The name of the target column.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.preprocessed_data = os.path.join(project_root, f"data/{competition_name}/prepocessed")
        self.feature_eng_data = os.path.join(
            project_root, f"data/{competition_name}/feature_engineered"
        )
        os.makedirs(self.feature_eng_data, exist_ok=True)
        self.target_column = target_column

    def stratified_sample(self, data, size_per_bin=1000, bins=15):
        """
        Perform stratified sampling on the data.

        Parameters:
        data (pd.DataFrame): The input data.
        size_per_bin (int): The size per bin for stratified sampling.
        bins (int): The number of bins for stratified sampling.

        Returns:
        tuple: Stratified samples, train and validation data.
        """
        data_s = data.copy()
        data_s["bins"] = pd.qcut(data_s[self.target_column], q=min(bins, len(data_s)//2), duplicates='drop', labels=False)

        sample = data_s.groupby("bins").apply(
            lambda x: x.sample(n=min(size_per_bin, len(x)), random_state=1)
        ).reset_index(drop=True)

        sample = sample.drop(columns=["bins"])
        X_train_stratified = sample.drop(columns=[self.target_column])
        y_train_stratified = sample[self.target_column]
        return X_train_stratified, y_train_stratified

    def openfe_fit(self, data, number_of_features=5, **kwargs):
        """
        Fit OpenFE on the data.

        Parameters:
        data (pd.DataFrame): The input data.
        number_of_features (int): The number of features to generate.

        Returns:
        OpenFE: The fitted OpenFE object.
        """
        ofe = OpenFE()
        X_train_stratified, y_train_stratified = self.stratified_sample(data, **kwargs)

        with contextlib.redirect_stdout(None):  # Suppress the output
            with warnings.catch_warnings():  # Suppress warnings
                warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.metrics")
                ofe.fit(data=X_train_stratified, label=y_train_stratified, n_jobs=4)

        print(f"The top {number_of_features} generated features are:")
        for feature in ofe.new_features_list[:number_of_features]:
            print(tree_to_formula(feature))

        return ofe
       
    def openfe_transform(self, train, test, number_of_features=5, **kwargs):
        """
        Transform the data using OpenFE.

        Parameters:
        data (pd.DataFrame): The input data.
        number_of_features (int): The number of features to generate.

        Returns:
        pd.DataFrame: The transformed data.
        """
        ofe = self.openfe_fit(train, number_of_features, **kwargs)
        # names = [tree_to_formula(f) for f in ofe.new_features_list[:number_of_features]]
        # apply re.sub(r'\W+', '_', name) to each name with apply and lambda
        train_transformed, test_transformed = transform(
            train, test, ofe.new_features_list[:number_of_features], n_jobs=4
        )

        # save the transformed data
        train_transformed.to_pickle(os.path.join(self.feature_eng_data, "train_transformed.pkl"))
        test_transformed.to_pickle(os.path.join(self.feature_eng_data, "test_transformed.pkl"))
        return train_transformed, test_transformed
