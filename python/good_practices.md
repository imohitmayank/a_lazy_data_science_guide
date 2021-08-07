Python Good Practices
=============

## Introduction

- Let's cover some of the good practices, which will not only help us to create a working but also a beautiful piece of code 😀
- To cover this topic, we will go through three parts, 
    1. Project structuring
    2. Code fomatting
    3. Additional tips

## Project Structuring

- In this part, we will basically talk about some good practices on how the complete python project can be structured. For this, we will look at three different possibilities, which anyone can choose based on how complex or advanced their project is going to be.

### Type 1: The Bare minimum

- This is the most basic and yet organised structuring someone can do. This could be used when our project will not have more than a few scripts. The directory of a sample project will look something like this *(folder structure from within `my_project` directory)*:

```{code-block}
my_project             # Root directory of the project
├── code               # Source codes
├── input              # Input files
├── output             # Output files
├── config             # Configuration files
├── notebooks          # Project related Jupyter notebooks (for experimental code)
├── requirements.txt   # List of external package which are project dependency
└── README.md          # Project README
```

- As obvious from the names, folder `code` contains the individual modules (`.py` files), `input` and `output` contains the input and output files respectively, `notebook` contains the `.ipynb` notebooks files we use for experimentation. Finally, `config` folder could contain parameters within `yaml` files and can be accessed by the code module files.
- `requirements.txt` contains a list of all external python packages needed by the project. One advantage of maintaining this file is that all of these packages can be easily installed using `pip install -r requirements.txt` command. *(No need of manually installing each and every external packages!)*. One example `requirements.txt` file is shown below *(with `package_name==package_version` format)*, 

```{code-block}
BeautifulSoup==3.2.0
Django==1.3
Fabric==1.2.0
Jinja2==2.5.5
PyYAML==3.09
Pygments==1.4
```

- Finally, `README.MD` contains the what, why and how of the project, with some idcode on how to run it.  

### Type 2: The favourite

- Overtime, the python community has unofficialy agreed on a structure which you will find on most of the python projects. It has been debated and beaten to death, and hence stood the requirement of the time. This structuring can be done when we want to better organize multiple groups of modules *(one for Database ETL, another for analytics, etc)*. Another advantage is that it becomes very easy to compile the code and create python projects.

### Type 3: Kedro

- Kedro is not an project structuring stargety, it's more of a python tool released by [QuantumBlack Labs](https://github.com/quantumblacklabs), which does project structuring for you :smile:. On top of it, they provide a pleathora of features to make our project oraganization and even code execution process super-easy, so that we can truly focus on what matters the most -- the coding!
- Their project structure is shown below. And btw, we can create a blank project by running `kedro new` command *(don't forget to install kedro first by `pip install kedro`)*

```
get-started         # Parent directory of the template
├── conf            # Project configuration files
├── data            # Local project data (not committed to version control)
├── docs            # Project documentation
├── logs            # Project output logs (not committed to version control)
├── notebooks       # Project related Jupyter notebooks (can be used for experimental code before moving the code to src)
├── README.md       # Project README
├── setup.cfg       # Configuration options for `pytest` when doing `kedro test` and for the `isort` utility when doing `kedro lint`
└── src             # Project source code
```

- While most of the directories are similar with other types, a few points should be noted. Kedro's way of grouping different modules is by creating different *"pipelines"*. These pipelines are present within `src` folder, which in turn contains the module files. Furthermore, they have clear segregation of individual functions which are executed - these are stored within `nodes.py` file *(within individual pipeline folder)*, and these functions are later connected with the input and output within `pipeline.py` file. Kedro also segregates the code and the parameters, by storing the parameters within `conf` folder. 
- Apart from just helping with organising the project, they also provide options for sequential or parallel executions. We can execute individual functions *(within `nodes.py`)*, or individual pipelines *(which are combination of functions)*, or the complete project at one go. We can also create doc of the complete project or compile and package the project as a python `.whl` file, with just a single command run. For more details, and believe me we have just touched the surface, refer their official [documentation](https://kedro.readthedocs.io/en/stable/index.html).

## Code formatting

- With a top down approach, let's first have a look at a neat piece of code. We will discuss individual ascepts of the code in more details later. For now, just assume if someone asks you to do some scripting, what an ideal piece of code file should look like.
- Following code is take from `csv_column_operations.py` module file. It was generated for the prompt: *"write a function which takes csv file as input and returns the sum of a column"*.

```{code-block} python
---
lineno-start: 1
---
"""Return sum of a column from CSV file

A module with "perform_column_sum" main function that computes and return sum 
of an user defined numerical column from an csv file passed as pandas dataframe

Author: Mohit Mayank <mohitmayank1@gmail.com>

Created: 4th August, 2021
"""

# imports
import pandas as pd # to handle csv files

# modules
def perform_column_sum(csv_file, column_to_perform_operation_on, operation_to_perform='sum'):
    """Performs numerical operation over a column of pandas dataframe

    Parameters
    -------------
    csv_file: pandas.DataFrame
        the csv file which contains the column on which operation is to be performed

    column_to_perform_operation_on: string
        the column name (must be present in the csv file)

    operation_to_perform: string
        which operation to be performed on the column. Supported: ['sum']

    Returns
    --------
    column_operation_result: numeric
        the result after applying numeric operation on csv column
    """
    # Step 1: data check and break
    # check 1: column should be present in the csv_file
    check_flag_col_presence = column_to_perform_operation_on in csv_file.columns
    # break the code if incorrect data is provided
    if not check_flag_col_presence:
        print(f"Column {column_to_perform_operation_on} is absent from the csv_file! Breaking code!")
        break

    # check 2: all values in the column should be of type numeric
    check_flag_data_type = pd.to_numeric(csv_file[column_to_perform_operation_on], errors='coerce').notnull().all()
    # break the code if incorrect data is provided
    if not check_flag_data_type:
        print(f"One or more values in column {column_to_perform_operation_on} is not numeric! Breaking code!")
        break

    # Step 2: extract the column
    column_data = csv_file[column_to_perform_operation_on]

    # Step 3: Perform the operation
    column_operation_result = None
    if operation_to_perform == 'sum':
        column_operation_result = sum(column_data)

    # Step 4: return the result
    return column_operation_result

# run when file is directly executed 
if __name__ == '__main__':
    # create a dummy dataframe
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'a', 'a']})
    # call the function
    answer = perform_column_sum(df, 'col1', 'sum')
    # match if the answer is correct
    assert(answer==6)
```

```{note}
Some might argue why do such an overkill for a simple piece of code. Note, it's a dummy example. In real life, you will develop lot more complex pieces of codes and hence it become quite important that we understand the gist.
```
- Now let's take a deeper dive into the individual aspect of the above code.

### Module structure

- A module is a python file with `.py` extension which contains the executable code or functions or classes, etc.
- Usually, we start the module with module definition, which is an area where we provide some basic details of the module. We can do so using follwing template *(and it can be easily compared to a real code shown above)*

```{code-block} python
"""<Short description>

<Long description>

Author: <Name> <email>

Created: <date>
"""
```

- Next, we should clearly segregate the parts of the module such as imports, code area, etc using comment lines.
- Finally, at the bottom we could include some examples on how to run the code. Including these scripts within `if __name__ == '__main__':` makes sure that they only run when the file is directly executed *(like `python csv_column_operations.py`)*. So these pieces of code doesnt run when you say import the module in another script.  

### Functions structure

- Functions are the basic block of code which perform a specific task. A module consist of several functions. To inform the user what a particular block of code does, we start the function with a function definition. A sample template is provided below, 

```{code-block}
"""Description

Paramters
---------
<parameter_1>: <data_type>
    <parameter_1_description>

Returns
---------
<output_1>: <data_type>
    <output_1_description>
"""
```
- After this we can start adding the relevant code lines. Make sure to separate different logical blocks of code within the functions using comments.
- One important thing to handle at the start of the coding section, is to check the parameters and input data for some datatype or data content related basic issues. A majority of silly mistakes happens when someone provides wrong input, in which case we should print or log warning message and gracefully exit. The above same code contains two such preliminary but important checks inside the step 1 section.

### Naming convention

- There are several fomatting conventions that we can follow, like [Camel Case](https://en.wikipedia.org/wiki/Camel_case), [Snake case](https://en.wikipedia.org/wiki/Snake_case), etc. It's quite subjective and depends with the develper. Below are some example of naming different entities of a python code *(taken from PIP8 conventions  - with some modifications )*:innocent:,
    - **Module name:** Modules should have short, all-lowercase names (ex: `csv_column_operations.py`)
    - **Function or method name:** Function names should be lowercase, with words separated by underscores as necessary to improve readability. Also, dont forget to add your verbs! (ex: `perform_column_sum()`)
    - **Variable name:** Similar to function name but without the verbs! (ex: `list_of_news`)
    - **Class name:** Class names should normally use the CapWords convention. (ex: `FindMax`)
    - **Constant name:** Constants are usually defined on a module level and written in all capital letters with underscores separating words. (ex: `MAX_OVERFLOW` and `TOTAL`).

### Add comments

- PEP-8 defines three types of comments, 
    - **Block comments:** which is written for a single or a collection of code lines. This can be done either when you want to explain a set of lines or just want to segregate code. In the above example, you can see `# Step {1, 2, 3}` used as segregation comment and `# run when file is directly executed` used to explain a set of code lines.
    - **Inline comments:** which are added on the same line as the code. For example, see how `# to handle csv files` is used to justify the pandas package import. PEP-8 suggests to use inline comments sparingly. 
    - **Documentation Strings:** these are used for documentations for module, functions or classes. PEP-257 suggests using multiline comment for docstring (using """). Example of module and function docstrings *(short for documentation strings)* is provided in the sample code above.
- We should be as descriptive about our comments as possible. Try to separate functional sections of your code, provide explanations for complex code lines, provide details about the input/output of functions, etc. How do you know you have enough comments? - If someone with half your expertise can understand the code without calling you middle of the night :triumph:   

### Indentations - Tabs vs Spaces

- Frankly, I am only going to touch this topic with a long stick :broom:. There are already several [articles](https://softwareengineering.stackexchange.com/questions/57/tabs-versus-spaces-what-is-the-proper-indentation-character-for-everything-in-e), [reddit threads](https://www.reddit.com/r/learnpython/comments/8cann8/tabs_vs_spaces_i_dont_get_it/) and even tv series (Silicon valley :tv:) where this topic has already been disussed a lot. 
- Want my 2 cents? Pick any modern IDE *(like VSCode, Sublime, etc)*, set indentations to tabs, and set 1 tab = 4 spaces. Done :smirk: 

## Additional tips

- Till now we have discussed how to either structure the project or format the code. Next, we will cover generic set of good practices which will save you some pain down the line :grimacing:

### Logging

- Instead of printing statement in the console which is temporary (do a `cls` and poof it's gone :dash:), a better idea is to save these statements in a separate file, which you can always go back and refer. This is logging :scroll:
- Python provides an [inbuilt function](https://docs.python.org/3.7/library/logging.html) for logging. BY refering the official [how to](https://docs.python.org/3.7/howto/logging.html), logging to a file is super easy, 

```{code-block} python
---
lineno-start: 1
---
# import
import logging
# config the logging behavior
logging.basicConfig(filename='example.log',level=logging.DEBUG)
# some sample log messages
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')
```
- Note, there is a hierarchical levels of logging to segregate different severity of logs. In the exmaple shown above, the `level` parameter denotes the minimal level that is tracked and hence saved to the file. As per the official [how to](https://docs.python.org/3.7/howto/logging.html) these are the different logging levels with some details about when to use which *(in increasing oder of severity)*, 

| Level | When it’s used |
|---|---|
| DEBUG | Detailed information, typically of interest only when diagnosing problems. |
| INFO | Confirmation that things are working as expected. |
| WARNING | An indication that something unexpected happened, or indicative of some problem in the near future (e.g. ‘disk space low’). The software is still working as expected. |
| ERROR | Due to a more serious problem, the software has not been able to perform some function. |
| CRITICAL | A serious error, indicating that the program itself may be unable to continue running. |

### Documentation

- Documentation of the code is an absolute must, if you are planning to maintain the code or hand it over to someone else in the foreseable future. Just ask any developer about their excitement on finding a readymade and well curated documentation for a package they are planning to use. On the other hand, it looks quite difficult to create one yourself, isn't it? I mean look at the beautiful docs of [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier) or [pandas](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html). :open_mouth:
- Well, sorry for the scare there, but actually it's quite simple :wink:. Remember all the function and module comment an the formatting we followed before, as it turn out we can leverage many open source tools like [pydoc](https://docs.python.org/3/library/pydoc.html) and [sphinx](https://www.sphinx-doc.org/en/master/usage/quickstart.html) to create full fledged html documentation! Going into details of how to exactly create the doc is out of scope of this article, but it is fairly easy to follow the "How to" steps of both the packages. 
- One last thing, if you are using Kedro, this process is even more simpler. All you have to do is run one command - `kedro build-docs --open` to create the documentations and utomatically open it in your default browser!

### Virtual environment

- Virtual environments (VE) can be thought of as the local copy of python environment, created specifically for a project. This local copy is like a blank slate, and required package needs to be installed here separately. It is extremely important to create a virtual environment for any new project, because, 
    - each project has its own depedency tree (one package of a specific verion needs another package of a specific version)
    - while developing a project we may need to downgrade or upgrade a package, which if done on the base python environment, will affect your system!
    - hence, a separate copy where you install whatever you want seems to be the most logical solution.
- Using VE basically requires two steps, 
    - **Create a VE:** this can be done by running command `python3 -m venv tutorial-env` at the project root directory.
    - **Activate VE:** this can be done by running command `tutorial-env\Scripts\activate.bat` on Windows and `source tutorial-env/bin/activate` on Unix of MacOS.
- And thats it! Install, uninstall, upgrade or downgrade whatever you want! 

```{note}
Remember to switch to another VE when you start working on another project or/and to deactivate the VE when you want to move to base VE.
```

## References

- [PEP8 Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)
- [Kedro](https://kedro.readthedocs.io/en/stable/index.html)
- [Python 3 documentation](https://docs.python.org/3/)