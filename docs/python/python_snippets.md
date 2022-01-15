Python Snippets
=============

## Add numpy array as Pandas column

- References from [stackoverflow answer](https://stackoverflow.com/questions/18646076/add-numpy-array-as-column-to-pandas-data-frame)

``` python linenums="1"
import numpy as np
import pandas as pd
import scipy.sparse as sparse

df = pd.DataFrame(np.arange(1,10).reshape(3,3))
arr = sparse.coo_matrix(([1,1,1], ([0,1,2], [1,2,0])), shape=(3,3))
df['newcol'] = arr.toarray().tolist()
print(df)
```

## Get members of python object

``` python linenums="1"
# import
import networkx as nx
from inspect import getmembers

# Fetching the name of all drawing related members of NetworkX class.
for x in getmembers(nx):
    if 'draw' in x[0]:
        print(x)
```

## Install packages using code

- References from [here](https://www.kaggle.com/getting-started/65975)

``` python linenums="1"
# import
import sys
import subprocess

# install package function
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# example
# install("pathfinding")
```

## Python packaging

``` python linenums="1"
# clean the previous build files
python setup.py clean --all
# build the new distribution files
python setup.py sdist bdist_wheel
# upload the latest version to pypi
twine upload --skip-existing dist/*
```

## Python virtual environment

``` python linenums="1"
# Create the virtual environment in the current directory
python -m venv projectnamevenv
# pick one of the below based on your OS (default: Linux)
# activate the virtual environment - Linux
.projectnamevenv\Scripts\activate
# activate the virtual environment - windows
# .\\projectnamevenv\\Scripts\\activate.bat
```

## Where is my Python installed?

- To know the exact location of where the python distribution is installed, follow the steps as suggested [here](https://stackoverflow.com/questions/647515/how-can-i-find-where-python-is-installed-on-windows)

``` python linenums="1"
import os
import sys
print(os.path.dirname(sys.executable))
```

## Find files or folders

- `glob` is a very efficient way to extract relevant files or folders using python.
- A few example are shown below.

``` python linenums="1"
# import
from glob import glob

# Ex 1: fetch all files within a directory
glob("../data/01_raw/CoAID/*")

# Ex 2: fetch all files within a directory with a pattern 'News*COVID-19.csv'
glob("../data/01_raw/CoAID/folder_1/News*COVID-19.csv")

# Ex 2: fetch all files within multiple directories which
#       follow a pattern 'News*COVID-19.csv'
glob("../data/01_raw/CoAID/**/News*COVID-19.csv")
```

## Increase the pandas column width in jupyter lab or notebook

- Most of the times, we have text in a dataframe column, which while displaying gets truncated. 
- One way to handle this to increase the max width of all columns in the dataframe (as shown below)

``` python linenums="1"
import pandas as pd
pd.set_option('max_colwidth', 100) # increase 100 to add more space for bigger text
```

## Parse date and time from string

- There are basically 2 ways to do this, (1) Trust machine 🤖: for majority of the 'famous' date writing styles, you can use `dateparser` package that automatically extracts the date and parse it into `datetime` format.

``` python linenums="1"
# import
from dateparser import parse
# parse
text = 'October 05, 2021'
dateparser.parse(text)
# output - datetime.datetime(2021, 10, 5, 0, 0)
```

which will return output ``.

- Another way is (2) DIY 💪: if you can create the [date time format](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes)), you can use `datetime` package directly.

``` python linenums="1"
# import
from datetime import datetime
# parse
text = 'October 05, 2021'
date_format = '%B %d, %Y'
datetime.strptime(text, date_format)
# output - datetime.datetime(2021, 10, 5, 0, 0)
```

## Bulk document insert in MongoDB

- While `pymongo` provides `insert_many` function for bulk insert, it breaks in case of duplicate key. We can handle it with following function, which in its worse case is similar to `insert_one`, but shines otherwise. 


``` python linenums="1"
# import
import pymongo
# function
def insert_many_wrapper(df, col):
    """bulk insert docs into the MongoDB while handling duplicate docs

    Parameters
        - df (pandas.dataframe): row as a doc with `_id` col
        - col (pymongo.db.col): pymongo collection object in which insertion is to be done
    """
    # make a copy and reset index
    df = df.copy().reset_index(drop=True)
    # vars
    all_not_inserted = True
    duplicate_count = 0
    ttl_docs = df.shape[0]
    # iterate till all insertions are done (or passed in case of duplicates)
    while all_not_inserted:
        # try insertion
        try:
            col.insert_many(df.to_dict(orient='records'), ordered=True)
            all_not_inserted = False
        except pymongo.errors.BulkWriteError as e:
            id_till_inserted = e.details['writeErrors'][0]['keyValue']['_id']
            index_in_df = df[df['_id']==id_till_inserted].index[0]
            print(f"Duplicate id: {id_till_inserted}, Current index: {index_in_df}")
            df = df.loc[index_in_df+1:, :]
            duplicate_count += 1
    # final status
    print(f"Total docs: {ttl_docs}, Inserted: {ttl_docs-len(duplicate_count)}, Duplicate found: {len(duplicate_count)}")
```

## Search top StackExchange questions

- Stack Exchange exposes several API endpoints to process the questions, answers or posts from their website. 
- A simple implementation to search and download the latest (from yesterday) and top voted questions is shown below. For more such API endpoints, consult their official [doc](https://api.stackexchange.com/docs). 

``` python linenums="1"
"""
Request StackExchange API to get the top 10 most voted 
    questions and their answer from yesterday
"""

import requests
import json
import datetime
import time

# Get the current date
today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)

# Get the current time
now = datetime.datetime.now()

# Get the time of yesterday
yesterday_time = now.replace(day=yesterday.day, month=yesterday.month, year=yesterday.year)

# Convert the time to epoch time
yesterday_epoch = int(time.mktime(yesterday_time.timetuple()))

# Get the time of today
today_time = now.replace(day=today.day, month=today.month, year=today.year)

# Convert the time to epoch time
today_epoch = int(time.mktime(today_time.timetuple()))

# Get the top 10 most voted questions and their answer from yesterday
url = "https://api.stackexchange.com/2.2/questions?pagesize=10&fromdate=" + \
    str(yesterday_epoch) + "&todate=" + str(today_epoch) + \
        "&order=desc&sort=votes&site=stackoverflow"

# Get the response from the API
response = requests.get(url)

# Convert the response to JSON
data = response.json()

# Print the data
print(json.dumps(data, indent=4))
```

## Export complete data from ElasticSearch

- Due to several memory and efficiency related limitations, it is non-trivial to export complete data from ElasticSearch database.
- That said, it is not impossible. PFB an `scan` based implementation that does the same for a dummy `test_news` index.

``` python linenums="1"
# import 
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from tqdm import tqdm

# config
index_name = 'test_news'
db_ip = 'http://localhost:9200'

# connect to elasticsearch
es = Elasticsearch([db_ip])

# fetch all data from elasticsearch
scroll = scan(es, index=index_name, query={"query": {"match_all": {}}})
data = []
for res in tqdm(scroll):
    data.append(res['_source'])

# convert to pandas dataframe and export as csv
pd.DataFrame(data).to_csv("news_dump.csv", index=False)
```

## Convert python literals from string 

- While I/O from database or config files, we may get some literals (ex list) in form of string, wherein they maintain their structure but the type. We can use `ast` package to convert them back to their correct type.
- Quoting the documentation, "With `ast.literal_eval` you can safely evaluate an expression node or a string containing a Python literal or container display. The string or node provided may only consist of the following Python literal structures: strings, bytes, numbers, tuples, lists, dicts, booleans, and None."

``` python linenums="1"
# import
import ast
# list literal in string format
list_as_string = '["a", "b"]'
# convert
list_as_list = ast.literal_eval(list_as_string) # Output: ["a", "b"]
```

## Conda cheat sheet

- [Conda](https://en.wikipedia.org/wiki/Conda_(package_manager)) an open-source, cross-platform, language-agnostic package manager and environment management system. Therein again we have multiple varieties, 
  - **Miniconda:** it's a minimilistic package with python, conda and some base packages.
  - **Anaconda:**  it's a bigger package with all the things in Miniconda plus around 150 high quality packages.

- While the complete documentation can be accessed from [here](https://docs.conda.io/projects/conda/en/latest/index.html), some important snippets are:

```python linenums="1"
# list all supported python versions
conda search python

# create a new conda environment (with new python version)
# note, py39 is the name of the env
conda create -n py39 python=3.9

# list all of the environments
conda info --envs

# activate an environment
conda activate py39 # where py39 is the name of the env

# deactivate the current environment
conda deactivate

# delete an environment
conda env remove -n py39
```
