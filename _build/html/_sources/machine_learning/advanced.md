Advanced ML topics
========================

- We will discuss some advanced topic in Machine Learning.

## Cross validation

- Cross validation is a technique in which at each iteration you create different split of train and dev data. At each such iteration, we train he model on the train split and validate on the remaining split. This way, event with small training data, we can perform multiple fold of validation.
- If you repeat this operation (for $N$ iterations) over the complete data such that (1) each data point belonged to the dev split at most once, (2) each data point belonged to train split $N-1$ times - its cross-validation.   
- I have used Stratified K-Folds cross-validator, you can use any function from the complete list mentioned here - [Sklearn Model selection](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection)

```{code-block} python
---
lineno-start: 1
---
# import =======
from sklearn.model_selection import StratifiedKFold

# code =============
# split the dataset into K fold test
def split_dataset(dataset, return_fold=0, n_splits=3, shuffle=True, random_state=1):
    """
    dataset: pandas dataframe
    return_fold: the fold out of `n_split` fold, which is to be returned
    n_splits: # cross fold
    """
    # defined the KFOld function
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    # defined the dataset
    X = dataset
    y = dataset['class'] # label/class

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        if return_fold == i:
            return dataset.loc[train_index], dataset.loc[test_index]

# example call
if __name__ == '__main__':
    # read the dataset
    df = pd.read_csv("....")
    # get one specific fold out of
    train, test = split_dataset(dataset=df, fold=0, n_splits=3)
    # run for all folds
    for fold in range(n_splits):
        train, test = split_dataset(dataset=df, fold=fold, n_splits=n_splits)
        # <perform actions here...>
```

## Hyper-parameter tuning

- Below is an example of hyperparameter tuning for SVR regression algorithm. There we specify the search space i.e. the list of algorithm parameters to try, and for each parameter combination perform a 5 fold CV test.
- More details: [Sklearn Hyperparameter tuning](https://scikit-learn.org/stable/modules/grid_search.html)
- More details: [Sklearn SVR Algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)

```{code-block} python
---
lineno-start: 1
---
# import =======
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error

# DATA LOAD ============
train_data = ...  # load the features and target on which to train

# SEARCH SPACE ============
search_space = [{'kernel': ['poly', 'rbf', 'sigmoid'],
               'C': [1, 10, 100], 'epsilon': [10, 1, 0.1, 0.2, 0.01]}]

# TUNING ============
scorer = make_scorer(mean_squared_error, greater_is_better=False)
svr_gs = GridSearchCV(SVR(), search_space, cv = 5, scoring=scorer, verbose=10, n_jobs=None)
svr_gs.fit(train_data['features'], train_data['target'])


# PRINT RESULT ============
parameter_result = []
print("Grid scores on training set:")
means = svr_gs.cv_results_['mean_test_score']
stds = svr_gs.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, svr_gs.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
    parameter_result.append({'mean': abs(mean), 'std': std, **params})

# SELECT BEST PARAMETERS ============
# select the settings with smallest loss
parameter_result = pd.DataFrame(parameter_result)
parameter_result = parameter_result.sort_values(by=['mean'])
best_settings = parameter_result.head(1).to_dict(orient='records')[0]

# FIT WITH BEST PARAMETERS ============
SVRModel = SVR(C=best_settings['C'],
            epsilon=best_settings['epsilon'],
            kernel= best_settings['kernel'])
SVRModel.fit(train_data['features'], train_data['target'])
```
