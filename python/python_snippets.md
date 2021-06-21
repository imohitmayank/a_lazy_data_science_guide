Python Snippets
=============

## Add numpy array as Pandas column

- References from [stackoverflow answer](https://stackoverflow.com/questions/18646076/add-numpy-array-as-column-to-pandas-data-frame)

```{code-block} python
---
lineno-start: 1
---
import numpy as np
import pandas as pd
import scipy.sparse as sparse

df = pd.DataFrame(np.arange(1,10).reshape(3,3))
arr = sparse.coo_matrix(([1,1,1], ([0,1,2], [1,2,0])), shape=(3,3))
df['newcol'] = arr.toarray().tolist()
print(df)
```

## Get members of python object

```{code-block} python
---
lineno-start: 1
---
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

```{code-block} python
---
lineno-start: 1
---
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

```{code-block} bash
---
lineno-start: 1
---
# clean the previous build files
python setup.py clean --all
# build the new distribution files
python setup.py sdist bdist_wheel
# upload the latest version to pypi
twine upload --skip-existing dist/*
```

## Python virtual environment

```{code-block} bash
---
lineno-start: 1
---
# Create and Activate Virtual environment
# -----------------------------------------
# Create the virtual environment in the current directory
python -m venv projectnamevenv
# pick one of the below based on your OS (default: Linux)
# activate the virtual environment - Linux
.projectnamevenv\Scripts\activate
# activate the virtual environment - windows
# .\\projectnamevenv\\Scripts\\activate.bat
```
