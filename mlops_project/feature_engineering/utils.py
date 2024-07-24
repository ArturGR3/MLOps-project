import contextlib
import time
import warnings
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from autogluon.core.metrics import make_scorer
from openfe import OpenFE, transform, tree_to_formula


def stratified_sample(data, target, size_per_bin, bins):
    """
    Perform stratified sampling on the given data based on the target variable.

    Parameters:
    data (pandas.DataFrame): The input data.
    target (str): The name of the target variable.
    size_per_bin (int): The number of samples to be selected from each bin.
    bins (int): The number of bins to divide the target variable into.

    Returns:
    X_train (pandas.DataFrame): The feature matrix of the stratified sample.
    y_train (pandas.Series): The target variable of the stratified sample.
    """
    # Make a copy of the data to avoid modifying the original data
    data = data.copy()
    # Create a new column 'bins' to store the bin labels based on the target variable
    data['bins'] = pd.qcut(data[target], q=bins, labels=False)
    # Perform stratified sampling by selecting 'size_per_bin' samples from each bin
    sample = data.groupby('bins', group_keys=False).apply(lambda x: x.sample(n=size_per_bin, random_state=1), include_groups=False)
    # Reset the index of the sampled data
    sample = sample.reset_index(drop=True)
    # Separate the feature matrix (X_train) and the target variable (y_train)
    X_train = sample.drop(columns=[target])
    y_train = sample[target]
    return X_train, y_train

# Function to train OpenFE (automated feature engineering)
def openfe_fit(X_train, y_train) -> OpenFE:
    """
    Fits the OpenFE model on the training data.

    Args:
        X_train (array-like): The input training data.
        y_train (array-like): The target training data.

    Returns:
        OpenFE: The fitted OpenFE model.
    """
    ofe = OpenFE()
    with contextlib.redirect_stdout(None):  # Suppress output
        with warnings.catch_warnings():  # Suppress warnings
            warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.metrics")
            ofe.fit(data=X_train, label=y_train, n_jobs=4)
    topk = 5  # Number of top features to print
    print(f'The top {topk} generated features are:')
    for feature in ofe.new_features_list[:topk]:
        print(tree_to_formula(feature))
    return ofe

def calculate_sample_bins(shares, train_df):
    """
    Calculates the sample bins based on the given parameters.

    Parameters:
    - shares (list): A list of sample shares to calculate the bins for.

    Returns:
    - sample_bins_zie (dict): A dictionary containing the sample bins for each value in the given list.
    """
    # Initialize an empty dictionary to store the sample bins
    sample_bins_zie = {}
    # Define the number of bins
    number_of_bins = 20 
    # Iterate over each share in the given list
    for share in shares:
        # Calculate the sample bin size based on the share and number of bins
        sample_bins_zie[share] = int(train_df.shape[0] * share/number_of_bins)
    # Return the dictionary of sample bins
    return sample_bins_zie

def estimate_time_and_features(sample_bins_zie, train_df, TARGET):
    """
    Estimates the time to fit OpenFE based on the size of the sample. It also returns the top features generated by OpenFE.

    Args:
        sample_bins_zie (dict): A dictionary containing the sample share and bin size.
        train_df (pandas.DataFrame): The training dataset.
        TARGET (str): The target variable.

    Returns:
        time_simulation (dict): A dictionary containing the sample share and the estimated time for OpenFE.
        top_features (dict): A dictionary containing the sample share and the top features.
    """
    # Initialize empty dictionaries to store the results
    time_simulation = {}
    top_features = {}

    # Iterate over the sample share and bin size dictionary
    for sample_share, bin_size in sample_bins_zie.items():
        # Perform stratified sampling on the training dataset
        X_train_fe, y_train_fe = stratified_sample(train_df, target=TARGET, size_per_bin=bin_size, bins=20)
        # Start the timer
        start = time.time()
        # Perform OpenFE
        ofe = openfe_fit(X_train_fe, y_train_fe)
        # Calculate the time taken for OpenFE and store it in the time_simulation dictionary
        time_simulation[sample_share] = round((time.time()-start)/60, 2)
        print(f"Time taken for {sample_share} sample share: {time_simulation[sample_share]} minutes")
        # Get the top 20 features from the OpenFE results and store them in the top_features dictionary
        top_features[sample_share] = [tree_to_formula(feature) for feature in ofe.new_features_list[:20]]

    # Return the time_simulation and top_features dictionaries
    return time_simulation, top_features, ofe

def calculate_hit_rate(top_features, baseline=0.6, topk=20):
    """
    Calculate the hit rate for each sample share based on the top features.

    Parameters:
    - top_features (dict): A dictionary containing the top features for each sample share.
    - baseline (float): The baseline sample share to compare against. Default is 0.35.
    - topk (int): The number of top features to consider. Default is 20.

    Returns:
    - hit_rate (dict): A dictionary containing the hit rate for each sample share.
    """

    hit_rate = {}
    for sample_share, features in top_features.items():
        hit_rate[sample_share] = len(set(top_features[baseline]).intersection(set(features))) / topk

    return hit_rate

# Function to get AutoGluon score
def get_AutoGluon_score(train, val, target, metric, preset='best_quality', time_min=5):
    """
    Calculates the AutoGluon score for a given train and validation dataset.

    Parameters:
    train (pandas.DataFrame): The training dataset.
    val (pandas.DataFrame): The validation dataset.
    target (str): The name of the target variable.
    metric (callable, optional): The evaluation metric to use. Defaults to rmsle.
    preset (str, optional): The preset configuration to use. Defaults to 'best_quality'.
    time_min (int, optional): The minimum time limit for training in minutes. Defaults to 5.

    Returns:
    tuple: A tuple containing the negative score and the feature importance.
    """
    
    # Create a TabularPredictor object with the specified label, evaluation metric, and verbosity level
    predictor = TabularPredictor(label=target, eval_metric=metric, verbosity=0)
    # Fit the predictor on the training data using the specified time limit, presets, and excluded model types
    predictor.fit(train_data=train, time_limit=time_min*60, presets=preset, excluded_model_types=['KNN', 'NN'])
    # Evaluate the predictor on the validation data and get the score
    score = predictor.evaluate(val)
    # Get the feature importance from the predictor
    feature_importance = predictor.feature_importance(val)
    # Get the name of the metric used for evaluation
    metric_name = list(score.keys())[0]
    # Return the negative score and the feature importance as a tuple
    return -score[metric_name], feature_importance

def replace_match(match):
    """
    Replaces characters in a string based on a predefined dictionary of replacements.

    Args:
        match (re.Match): The matched character to be replaced.

    Returns:
        str: The replacement string.

    Raises:
        KeyError: If the matched character is not found in the replacements dictionary.
    """
    replacements = {
        '.': '_',
        '(': '_',
        ')': '_',
        ' ': '_',
        ',': 'AND',
        '+': 'PLUS',
        '-': 'MINUS',
        '*': 'TIMES',
        '/': 'DIVIDED'
    }
    return replacements[match.group(0)]