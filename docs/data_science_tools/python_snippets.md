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

## Get list of installed Python packages

- To know exactly which packages are current installed (and their version) in your VE, try

``` shell linenums="1"
pip list --format=freeze > reqirements.txt
```

## Unzipping files using Zipfile

``` python linenums="1"
# import
import zipfile
# unzipping
with zipfile.ZipFile('sample.zip', 'r') as zip_ref:
    zip_ref.extractall()
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

## Plotly visualization on Pandas dataframe

- If you want to visualize your pandas dataframe using plotly package, there is no need to use the package explicitly. It can be done right from the pandas dataframe object, with just a couple of lines of code as shown below:

```python linenums="1"
# set the backend plotting option
pd.options.plotting.backend = "plotly"
# do a normal plot!
pd.DataFrame(result).plot(x='size', y='mean')
```

## Conda cheat sheet

- [Conda](https://en.wikipedia.org/wiki/Conda_(package_manager)) an open-source, cross-platform, language-agnostic package manager and environment management system. Therein again we have multiple varieties, 
  - **Miniconda:** it's a minimilistic package with python, conda and some base packages.
  - **Anaconda:**  it's a bigger package with all the things in Miniconda plus around 150 high quality packages.

- While the complete documentation can be accessed from [here](https://docs.conda.io/projects/conda/en/latest/index.html), some important snippets are:

```python linenums="1"
# list all supported python versions
conda search python

# create a new global conda environment (with new python version)
# note, py39 is the name of the env
conda create -n py39 python=3.9

# create a new local conda environment 
# (under venv folder in current directory and with new python version)
conda create -p ./venv -n py39 python=3.9

# list all of the environments
conda info --envs

# activate an environment
conda activate py39 # where py39 is the name of the env

# deactivate the current environment
conda deactivate

# delete an environment
conda env remove -n py39
```

## uv cheat sheet

- [uv](https://github.com/astral-sh/uv) is an extremely fast Python package installer and resolver written in Rust. It serves as a drop-in replacement for pip, pip-tools, poetry, pyenv, twine, virtualenv, and more.

### Creating projects

```shell linenums="1"
# Initialize a project in the current directory
uv init

# Initialize a project in a specific directory
uv init project_name

# Specify Python version for the project
uv init --python 3.11
```
### Virtual environments

```shell linenums="1"
# Create virtual environment
uv venv .venv

# Use uv as faster pip replacement
uv pip install package_name

# Create virtual environment with specific Python version
uv venv --python 3.11
```

### Managing dependencies

```shell linenums="1"
# Add a package as dependency
uv add pandas

# Add multiple packages at once
uv add numpy scipy matplotlib

# Add dependencies from requirements file
uv add -r requirements.txt

# Add development dependencies
uv add --dev pytest black

# Run a command from installed packages
uv run pytest

# Remove multiple dependencies
uv remove numpy scipy

# View dependency tree
uv tree

# Upgrade all dependencies
uv lock --upgrade
```

### Version management

```shell linenums="1"
# Check current project version
uv version
```

### Building and publishing

```shell linenums="1"
# Build your package
uv build

# Publish to PyPI
uv publish
```

### Working with scripts

```shell linenums="1"
# Initialize a standalone script
uv init --script script.py

# Initialize script with specific Python version
uv init --script script.py --python 3.11

# Add dependency to a script
uv add requests --script script.py

# Run a script
uv run script.py

# Run script with specific Python version
uv run --python 3.11 script.py

# Run script with specific Python version and dependencies
uv run --python 3.11 --with torch,peft==0.18.0rc0 test_model_size.py

# Run Python interpreter with specific Python version and dependencies
uv run --python 3.11 --with torch,peft==0.18.0rc0 python -c "import sys; print(sys.version)"

# Run script with specific Python version and dependencies from requirements file
uv run --with-requirements requirements.txt crawl.py
```

### Python version management

```shell linenums="1"
# List available Python versions
uv python list

# Install a Python version
uv python install 3.11

# Run Python interpreter
uv run python

# Run specific Python version
uv run --python 3.11 python
```

### Code formatting

```shell linenums="1"
# Format code with Ruff
uv format
```

## Requirement files

- Requirement file is a collection of packages you want to install for a Project. A sample file is shown below, 

``` shell
# fine name requirements.txt
package-one==1.9.4
git+https://github.com/path/to/package-two@41b95ec#egg=package-two
package-three==1.0.1
package-four
```
- Note three ways of defining packages, (1) with version number, (2) with github source and (3) without version number (installs the latest). Once done, you can install all these packages at one go by `pip install -r requirements.txt`

## Reading `.numbers` file

- `.numbers` file is a proprietary file format of Apple's Numbers application. It is a spreadsheet file format that is used to store data in a table format. To process and load the data from `.numbers` file, we can use [numbers_parser](https://github.com/masaccio/numbers-parser) package. Below is an example of how to read the data from multiple `.numbers` files and combine them into one file.

```python linenums="1"
# import
import os
import glob
import pandas as pd
from tqdm import tqdm
from numbers_parser import Document

# Get the list of files in the directory using glob
numbers_files = glob.glob(cwd + '/*.numbers')

# Combine the .numbers files into one file
combined_df = []
for file in tqdm(numbers_files):
    doc = Document(file)
    sheets = doc.sheets
    tables = sheets[0].tables
    data = tables[0].rows(values_only=True)
    df = pd.DataFrame(data[1:], columns=data[0])
    df['file'] = file
    # Append the dataframe to the combined dataframe
    combined_df.append(df)
    
# Save the combined file
combined_df = pd.concat(combined_df)
combined_df.to_csv('combined_numbers_new.csv', index=False)
```

## Pandas Groupby Function

- Pandas can be utilised for fast analysis of categorical data using groupby. Let's have a look.

```python linenums="1"
#import
import numpy as np
import pandas as pd

# load a dummy df
df = pd.Dataframe('dummy.csv') # example below
## Name | Gender   | Salary
## Ravi | Male     | $20,000
## Sita | Female   | $40,000
## Kito | Female   | $11,000

# perform groupby to get average salary per gender
## Option 1
df.groupby(['Gender']).agg({'Salary': [np.mean]})
## Option 2
df.groupby(['Gender']).mean()
## Option 3
df.groupby(['Gender']).apply(lambda x: x['Salary'].mean())
```

## Save and Load from Pickle

- Pickle can be used to efficiently store and load python objects and data. Refer [StackOverflow](https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict-or-any-other-python-object)

```python linenums="1"
# import
import pickle

# create data
a = {'a': 1, 'b': [1,2,3,4]}

# save pickle
with open('filename.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load pickle
with open('filename.pickle', 'rb') as handle:
    b = pickle.load(handle)

# check
assert print(a == b)
```

## Download Youtube video

- Youtube video can be downloaded using the `pytube` package. Here is an example.

```python linenums="1"
# import
from pytube import YouTube

## var: link to download
video_url = "https://www.youtube.com/watch?v=JP41nYZfekE"

# create instance
yt = YouTube(video_url)

# download
abs_video_path = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()
## print(f"Video downloaded at {abs_video_path}")    
```

## Machine Translation

- EasyNMT lets you perform state-of-the-art machine translation with just 3 lines of python code!
- It supports translation between 150+ languages and automatic language detection for 170+ languages. Pre-trained machine translation models are auto-downloaded and can perform sentence and document translations!

```python linenums="1"
# import
from easynmt import EasyNMT

# load model
model = EasyNMT('opus-mt')

#Translate a single sentence to German
print(model.translate('This is a sentence we want to translate to German', target_lang='de'))
## Output: Dies ist ein Satz, den wir ins Deutsche übersetzen wollen
```

## Pandas read excel file

- While `pandas` is quite famous for CSV analysis, it can be used to read and process Excel files as well. Here are some snippets, 

```python linenums="1"
# import
import pandas as pd

# if you just want to read one sheet, by default it reads the first one. 
df = pd.read_excel("file.xlsx", sheet_name="Page1")

# if you want to get the names of sheet and do more selective reading
excel_data = pd.ExcelFile("file.xlsx")
# get the sheet names
print(excel_data.sheet_names)
# read one sheet (decide using last print result)
sheet_name = '..' 
df = excel_data.parse(sheet_name)
```

## Send Slack Messages

- One of the easiest way to send Slack message is via unique Incoming Webhook. 
- Basically, you need to create a Slack App, register an incoming webhook with the app and whenever you want to post a message - just send a payload to the webhook. For more details on setup, you can refer the [official page](https://api.slack.com/messaging/webhooks)
- Once done, you just need to send the message like shown below, 

```python linenums="1"
# import requests (needed to connect with webhook)
import requests
# func
def send_message_to_slack(message):
    # set the webhook
    webhook_url = "...enter incoming webhook url here..."
    # modify the message payload
    payload = '{"text": "%s"}' % message
    # send the message
    response = requests.post(webhook_url, payload)
# test
send_message_to_slack("test")
```

## Colab Snippets

- [Google Colab](https://colab.research.google.com/) is the go-to place for many data scientists and machine learning engineers who are looking to perform quick analysis or training for free. Below are some snippets that can be useful in Colab.

- If you are getting `NotImplementedError: A UTF-8 locale is required. Got ANSI_X3.4-1968` or similar error when trying to run `!pip install` or similar CLI commands in Google Colab, you can fix it by running the following command before running `!pip install`. But note, this might break some imports. So make sure to import all the packages before running this command.

```python linenums="1"
import locale
locale.getpreferredencoding = lambda: "UTF-8"

# now import
# !import ...
```

<!-- ## Python Classmethod vs Staticmethod

https://stackoverflow.com/questions/12179271/meaning-of-classmethod-and-staticmethod-for-beginner -->


## Asyncio Gather

- `asyncio.gather` is a powerful function in Python's `asyncio` module that allows you to run multiple coroutines concurrently and collect the results. Here is an example of how to use `asyncio.gather` to process a list of inputs concurrently and maintain order.

```python linenums="1"
# import 
import asyncio
import random

# func to process input
async def process_input(input_value):
    # Generate a random sleep time between 1 and 5 seconds
    sleep_time = random.uniform(1, 5)
    print(f"Processing {input_value}. Will take {sleep_time:.2f} seconds.")
    
    # Simulate some processing time
    await asyncio.sleep(sleep_time)
    
    return f"Processed: {input_value}"

async def main():
    # List of inputs to process
    inputs = ["A", "B", "C", "D", "E"]
    
    # Create a list of coroutines to run
    tasks = [process_input(input_value) for input_value in inputs]
    
    # Use asyncio.gather to run the coroutines concurrently and maintain order
    results = await asyncio.gather(*tasks)
    
    # Print the results
    for input_value, result in zip(inputs, results):
        print(f"Input: {input_value} -> {result}")

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
```

- Once you run the above code, here is an example of the output you might see:

```shell
Processing A. Will take 2.45 seconds.
Processing B. Will take 1.47 seconds.
Processing C. Will take 4.47 seconds.
Processing D. Will take 1.68 seconds.
Processing E. Will take 4.47 seconds.
Input: A -> Processed: A
Input: B -> Processed: B
Input: C -> Processed: C
Input: D -> Processed: D
Input: E -> Processed: E
```

- As you can see from the output, the inputs are processed concurrently, and the results are collected in the order of the inputs. This is true even if the processing times are different for each input.

