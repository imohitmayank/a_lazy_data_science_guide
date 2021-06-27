Basic ML topics
========================

- We will go through some basic ML topics.

## Feature importance

- Feature importance calculation is an important technique to identify the  features which "helps" with the downstream classification or regression tasks. 
- Sklearn provides several options to infer the importance of a feature. Most importantly, many model automatically computed the importane and store it in `model.feature_importances_`, after you call `.fit()` 
- As an example, lets take the text based classification task and try to infer the following, 
    - **Part 1:** First use `CountVectorizer` for feature engineering and `ExtraTreesClassifier` for classification. 
    - **Part 2:** Show the top N features.
    - **Part 3:** Show evidence of a feature (by value count over different class labels)
- Following dataset based assumtions have been made,  
    - We assume `x_train` and `y_train` contains the a list of sentences and labels repectively.
    - We assume a pandas dataframe of name `train_df` is present which contains `x_train` and `y_train` as columns with name `title` and `label` respectively. 

```{code-block} python
---
lineno-start: 1
---
# import
import random
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer

# PART 1: Train the model
# -----------------------
# variables
max_features = 10000

# get features
vectorizer = CountVectorizer(max_features=max_features)
features = vectorizer.fit_transform(x_train)

# model
model = ExtraTreesClassifier(random_state=1)
model.fit(features, y_train)


# PART 2: View top features
# -----------------------

top_n = 10 # no of top features to extract 
feature_imp_indices = model.feature_importances_.argsort()[-top_n:][::-1]
feature_importance = pd.DataFrame({'score': model.feature_importances_[feature_imp_indices], 
                                  'feature': np.array(vectorizer.get_feature_names())[feature_imp_indices],
                                  'indices': feature_imp_indices})
feature_importance # a pandas dataframe of top_n features


# PART 3: View individual feature's evidence
# -----------------------

index = 2282 # the feature's index 
# the label's distribution if this word is present in sentence
train_df.iloc[np.where(features[:, index].todense() >= 1)[0]]['label'].value_counts()

````