Python Good Practices
=============

## Introduction

- Writing code that works now is easy. Writing code that will work tomorrow is hard. Writing code that will work tomorrow and is intuitive enough for anyone to understand and follow — well now we have hit the super hard stuff 😀. Observing several ML engineers and Data scientists working with me, I have noticed nearly all of them have their own unique style of coding. Well, don't get me wrong, subjectively is a good thing and I think it is what leads to innovations. That said while working in a team or even in open source collaboration, it helps to agree to a certain set of rules. And that's the idea behind this article, to provide python practitioners with a set of curated guidelines, from which they can pick and choose.
- With that, let's cover some of the good practices, which will not only help us to create a working but also a beautiful piece of code 😀
- To cover this topic, we will go through three parts, 
  1. **Project structuring:** ideas on how to organize your code
  2. **Code formatting:** ideas on how to make your code easy to follow
  3. **Additional tips:** a few things which will help you in the longer run

## Project Structuring

- In this part, we will basically talk about some good practices on how the complete python project can be structured. For this, we will look at two different possibilities, which anyone can choose based on how simple or complex their project is going to be.

### Type 1: The Classic

- This is the most basic format and yet gives the hint of organized structuring. This can be followed when our project consists of only a few scripts. The directory of a sample project could look something like this:

``` python linenums="1"
my_project             # Root directory of the project
├── code               # Source codes
├── input              # Input files
├── output             # Output files
├── config             # Configuration files
├── notebooks          # Project related Jupyter notebooks (for experimental code)
├── requirements.txt   # List of external package which are project dependency
└── README.md          # Project README
```

- As obvious from the names, folder `code` contains the individual modules (`.py` files), `input` and `output` contains the input and output files respectively, and `notebook` contains the `.ipynb` notebooks files we use for experimentation. Finally, `config` folder could contain parameters within `yaml` or `json` or `ini` files and can be accessed by the code module files using [configparser]([configparser — Configuration file parser &#8212; Python 3.7.11 documentation](https://docs.python.org/3.7/library/configparser.html)).
- `requirements.txt` contains a list of all external python packages needed by the project. One advantage of maintaining this file is that all of these packages can be easily installed using `pip install -r requirements.txt` command. *(No need of manually installing each and every external packages!)*. One example `requirements.txt` file is shown below *(with `package_name==package_version` format)*, 

``` python linenums="1"
BeautifulSoup==3.2.0
Django==1.3
Fabric==1.2.0
Jinja2==2.5.5
PyYAML==3.09
Pygments==1.4
```

- Finally, `README.MD` contains the what, why and how of the project, with some dummy codes on how to run the project and sample use cases.  

### Type 2: Kedro

- Kedro is not a project structuring strategy, it's a python tool released by [QuantumBlack Labs](https://github.com/quantumblacklabs), which does project structuring for you 😎. On top of it, they provide a plethora of features to make our project organization and even code execution process super-easy, so that we can truly focus on what matters the most -- the experimentations and implementations!
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

- While most of the directories are similar to other types, a few points should be noted. Kedro's way of grouping different modules is by creating different *"pipelines"*. These pipelines are present within `src` folder, which in turn contains the module files. Furthermore, they have clear segregation of individual functions which are executed - these are stored within `nodes.py` file, and these functions are later connected with the input and output within `pipeline.py` file *(all within the individual pipeline folder)*. Kedro also segregates the code and the parameters, by storing the parameters within `conf` folder. 
- Apart from just helping with organizing the project, they also provide options for sequential or parallel executions. We can execute individual functions *(within `nodes.py`)*, or individual pipelines *(which are a combination of functions)*, or the complete project at one go. We can also create doc of the complete project or compile and package the project as a python `.whl` file, with just a single command run. For more details, and believe me we have just touched the surface, refer to their official [documentation](https://kedro.readthedocs.io/en/stable/index.html).

## Code formatting

- With a top-down approach, let's first have a look at a *neat* piece of code. We will discuss individual aspects of the code in more detail later. For now, just assume if someone asks you to do some scripting, what an ideal piece of code file should look like.
- Following code is take from `csv_column_operations.py` module file. It was generated for the prompt: *"write a function which takes csv file as input and returns the sum of a column"*.

``` python linenums="1"
"""Return sum of a column from CSV file

A module with "perform_column_sum" main function that computes and return sum 
of an user defined numerical column from an csv file passed as pandas dataframe

Author: Mohit Mayank <mohitmayank1@gmail.com>

Created: 4th August, 2021
"""

# imports
import sys          # to exit execution
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
        sys.exit()

    # check 2: all values in the column should be of type numeric
    check_flag_data_type = pd.to_numeric(csv_file[column_to_perform_operation_on], errors='coerce').notnull().all()
    # break the code if incorrect data is provided
    if not check_flag_data_type:
        print(f"One or more values in column {column_to_perform_operation_on} is not numeric! Breaking code!")
        sys.exit()

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

!!! note
    Some might argue why do such an overkill for a simple piece of code. Note, it's a dummy example. In real life, you will develop more complex pieces of codes and hence it become quite important that we understand the gist.

- Now let's take a deeper dive into the individual aspect of the above code.

### Module structure

- A module is a python file with `.py` extension that contains the executable code or functions or classes, etc.
- Usually, we start the module with module definition, which is an area where we provide some basic details of the module. We can do so using the following template *(and it can be easily compared to a real code shown above)*

``` python linenums="1"
"""<Short description>

<Long description>

Author: <Name> <email>

Created: <date>
"""
```

- Next, we should clearly segregate the parts of the module such as imports, code area, etc using comment lines.
- Finally, at the bottom, we could include some examples on how to run the code. Including these scripts within `if __name__ == '__main__':` makes sure that they only run when the file is directly executed *(like `python csv_column_operations.py`)*. So these pieces of code doesn't run when you say import the module in another script.  

### Functions structure

- Functions are the basic block of code that performs a specific task. A module consists of several functions. To inform the user what a particular block of code does, we start the function with a function definition. A sample template is provided below, 

``` python linenums="1"
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

- After this, we can start adding the relevant code lines. Make sure to separate different logical blocks of code within the functions using comments.
- One important thing to handle at the start of the coding section is to check the parameters and input data for some data type or data content related basic issues. A majority of code break happens due to silly mistakes like when someone provides wrong input, in which case we should print or log warning message and gracefully exit. The above same code contains two such preliminary but important checks inside the step 1 section.

### Naming convention

- There are several formatting conventions that we can follow, like [Camel Case](https://en.wikipedia.org/wiki/Camel_case), [Snake case](https://en.wikipedia.org/wiki/Snake_case), etc. It's quite subjective and depends on the developer. Below are some examples of naming different entities of a python code *(taken from PIP8 conventions  - with some modifications)* 😇,
  - **Module name:** Modules should have short, all-lowercase names (ex: `csv_column_operations.py`)
  - **Function or method name:** Function names should be lowercase, with words separated by underscores as necessary to improve readability. Also, don't forget to add your verbs! (ex: `perform_column_sum()`)
  - **Variable name:** Similar to function name but without the verbs! (ex: `list_of_news`)
  - **Class name:** Class names should normally use the CapWords convention. (ex: `FindMax`)
  - **Constant name:** Constants are usually defined on a module level and written in all capital letters with underscores separating words. (ex: `MAX_OVERFLOW` and `TOTAL`).

### Add comments

- PEP-8 defines three types of comments, 
  - **Block comments:** which is written for a single or a collection of code lines. This can be done either when you want to explain a set of lines or just want to segregate code. In the above example, you can see `# Step {1, 2, 3}` used as segregation comment and `# run when file is directly executed` used to explain a set of code lines.
  - **Inline comments:** which are added on the same line as the code. For example, see how `# to handle csv files` is used to justify the pandas package import. PEP-8 suggests using inline comments sparingly. 
  - **Documentation Strings:** these are used for documentation for module, functions or classes. PEP-257 suggests using multiline comment for docstring *(using """)*. An example of module and function docstrings *(short for documentation strings)* is provided in the sample code above.
- We should be as descriptive in our comments as possible. Try to separate functional sections of your code, provide explanations for complex code lines, provide details about the input/output of functions, etc. How do you know you have enough comments? - If you think someone with half your expertise can understand the code without calling you middle of the night! 😤   

### Indentations - Tabs vs Spaces

- Frankly, I am only going to touch this topic with a long stick 🧹. There are already several [articles](https://softwareengineering.stackexchange.com/questions/57/tabs-versus-spaces-what-is-the-proper-indentation-character-for-everything-in-e), [reddit threads](https://www.reddit.com/r/learnpython/comments/8cann8/tabs_vs_spaces_i_dont_get_it/) and even tv series (Silicon valley 📺) where this topic has been discussed a lot!
- Want my 2 cents? Pick any modern IDE *(like VSCode, Sublime, etc)*, set indentations to tabs, and set 1 tab = 4 spaces. Done 😏

## Additional tips

- Till now we have discussed how to either structure the project or format the code. Next, we will cover a generic set of good practices which will save you some pain down the line 😬

### Logging

- Instead of printing statements in the console which is temporary (do a `cls` and poof it's gone💨), a better idea is to save these statements in a separate file, which you can always go back and refer to. This is logging 📜
- Python provides an [inbuilt function](https://docs.python.org/3.7/library/logging.html) for logging. By referring to the official [how to](https://docs.python.org/3.7/howto/logging.html), logging to a file is super easy, 

``` python linenums="1"
# import
import logging
# config the logging behavior
logging.basicConfig(filename='example.log',level=logging.DEBUG)
# some sample log messages
logging.debug('This message should go to the log file', exc_info=True)
logging.info('So should this', exc_info=True)
logging.warning('And this, too', exc_info=True)
# we add "exc_info=True" to capture the stack trace
```

- Note, there is a hierarchical levelling of logs to segregate different severity of logs. In the example shown above, the `level` parameter denotes the minimal level that is tracked and hence saved to the file. As per the official [how to](https://docs.python.org/3.7/howto/logging.html), these are the different logging levels with some details about when to use which *(in increasing order of severity)*, 

| Level    | When it’s used                                                                                                                                                         |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DEBUG    | Detailed information, typically of interest only when diagnosing problems.                                                                                             |
| INFO     | Confirmation that things are working as expected.                                                                                                                      |
| WARNING  | An indication that something unexpected happened, or indicative of some problem in the near future (e.g. ‘disk space low’). The software is still working as expected. |
| ERROR    | Due to a more serious problem, the software has not been able to perform some function.                                                                                |
| CRITICAL | A serious error, indicating that the program itself may be unable to continue running.                                                                                 |

- While the above code is good for normal testing, for production you might want to have more control -- like formatting the output slightly differently *(formatter)* or have multiple places to publish logs *(handlers)*. One such use case is convered below, where we want to log to console as well as a file in a detailed json format.

```python
# import
import sys
import logging
import json_log_formatter

# create formatter (using json_log_formatter)
formatter = json_log_formatter.VerboseJSONFormatter()

# create two handlers (console and file)
logger_sys_handler = logging.StreamHandler(sys.stdout)
logger_file_handler = logging.FileHandler(filename='log.json')

# perform the same formatting for both handler
logger_sys_handler.setFormatter(formatter)
logger_file_handler.setFormatter(formatter)

# get the logger and add handlers
logger = logging.getLogger('my_logger')
logger.addHandler(logger_sys_handler)
logger.addHandler(logger_file_handler)

# set level
logger.setLevel(logging.INFO)
```

### Documentation

- Documentation of the code is an absolute must, if you are planning to maintain the code or hand it over to someone else in the foreseeable future. Just ask any developer about their excitement on finding a ready-made and well curated documentation for a package they are planning to use! On the other hand, it looks quite difficult to create one yourself, isn't it? I mean, look at the beautiful docs of [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier) or [pandas](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html). 😮
- Well, sorry for the scare there, but actually it's quite simple 😉. Remember all the function and module docstring and the formatting we followed before? As it turns out, we can leverage many open source tools like [pydoc](https://docs.python.org/3/library/pydoc.html) and [sphinx](https://www.sphinx-doc.org/en/master/usage/quickstart.html) to create full-fledged HTML documentations! Going into practical details is out of scope of this article, but it is fairly easy to follow the "how to" steps of both the packages and get your doc ready.
- One last thing, if you are using Kedro, this process is even simpler. All you have to do is run one command - `kedro build-docs --open` to create the documentation and automatically open it in your default browser!

### Virtual environment

- Virtual environments (VE) can be thought of as the local copy of python environment, created specifically for a project. This local copy is like a blank slate, as any required package can be installed here separately. It is extremely important to create a new virtual environment for any new project, because, 
  - each project has its own dependency tree *(one package with a specific version needs another package to run, with its own specific version)*
  - while developing a project we may need to downgrade or upgrade a package, which if done on the base python environment, will affect your system!
  - hence, a separate copy of python (VE), where you install whatever you want, seems to be the most logical solution.
- Using VE basically requires two steps, 
  - **Create a VE:** this can be done by running command `python3 -m venv tutorial-env` at the project root directory. *(note, `tutorial-env` is the name of the VE, you can use rename it to anything)*
  - **Activate VE:** this can be done by running command `tutorial-env\Scripts\activate.bat` on Windows and `source tutorial-env/bin/activate` on Unix or MacOS.
- And that's it! Install, uninstall, upgrade or downgrade whatever you want! 

!!! note
    Remember to switch to another VE when you start working on another project or/and to deactivate the VE when you want to move to base VE.

## References

- [PEP8 Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)
- [Kedro](https://kedro.readthedocs.io/en/stable/index.html)
- [Python 3 documentation](https://docs.python.org/3/)