# Engineering

- Several aspiring data scientists think Data science is just about training fancy models that does some superficial stuff. While it's part of the job, what we are missing here is that majority of the work is required to make sure that a model is trainable, executable and deployable. Add to it the complexity of working in a team and hence making sure the code is well formatted and structured. In all, life of a Data scientist is similar to any software engineer, just with a caveat of having the luxury to play with the state-of-the-art AI algorithms once in a while 😄

- Now, the industry is trying (or realizing) the capabilities, limitations and responsibilities of professionals in the AI or ML field. This is giving rise to increase in requirements for diverse job profiles like ML engineer, MLOps engineer, Data Scientist, Research Scientist, etc. That said, Data scientist (DS) are reinventing themselves as well, giving rise to the interesting profile of Full stack DS - who while researching and experiment with AI models, is not afraid to dabble into the old school engineering aspect of the project. This article is for such aspirants or practitioners.

- Let's go through some of the "engineering" aspects of the project that will save you wish you knew at the start of the project.

## Version Control

- Version control (VC) is basically needed for any file(s) that will be maintained for a long time *(read, multiple editing process)* and/or accessed by multiple peoples *(in collaboration)*. If you have such a file or a set of file *(as a project)*, you will agree the necessity to track the changes as they happen. And VC tries to help us with the same 🆒

- Now comes the question of what can be tracked and how? A very simple distinction of different tools available for VC can be introduced efficiently, if we look at them from the lens of what type of data they can "version control". Such as, 
  
  - **Code:** if you want to maintain just the code files, GIT is the defacto answer. There are several GIT based service providers whose platform can be used *(for free)* to maintain a git repository, such as "Github", "Gitlab", "BitBucket", etc.
  
  - **Data and ML Model:** in contrast to GIT, which was developed to maintain relatively small sized files, we need something different if we want to handle big files like datasets or ML/DL models. Enter [DVC](https://dvc.org/) (Data Version Control), an GIT extension that directly connects with data storages (cloud or local) with the git to maintain data, model and code at the same time!

- We will go through some of the tools/services in little detail.

### GIT

#### Introduction

- GIT is the defacto version control for code base, with popular free and services like GitHub, Gitlab and BitBucket.
- TODO

#### GIT Snippets

- Listing down some of the helper code snippets for GIT

##### Modify config to add email and name

```{code-block}
---
lineno-start: 1
---
# Add username
git config --global user.name "FIRST_NAME LAST_NAME"
# Add email
git config --global user.email "MY_NAME@example.com"

# For local modification (for a git within a directory), use --local instead of --global
# mode details: https://support.atlassian.com/bitbucket-cloud/docs/configure-your-dvcs-username-for-commits/
```

##### Ignore files/folders

- `.gitignore` file in the root directory, contains the name of files and folders which should not be tracked by GIT.

```{code-block}
# ignore personal.txt
personal

# ignore everything within "pic" folder
pic/*

# ignore everything except a specific file within a folder
!pic/logo.png
pic/*
```

##### Untrack file/folder and delete them from GIT

- To untrack the files or folders, we can create `.gitignore` file and add respective info.
- To delete the files or folders form GIT (and not from local system), we can delete them from the cache as suggested [here](https://stackoverflow.com/questions/1143796/remove-a-file-from-a-git-repository-without-deleting-it-from-the-local-filesyste),

```{code-block}
# for a single file:
git rm --cached mylogfile.log

# for a single directory:
git rm --cached -r mydirectory
```

##### Stash partial changes

- Suppose you have made some partial changes and the remote is updated with a new commit. Now you cannot commit your local change (as its partial) and you need to pull the latest code from remote (as its update). `git stash` comes to the rescue, example below.

```{code-block}
# stash away the  current partial changes
git stash

# pull the latest code (or any other operation)
git pull origin master

# pop the stashed partial changes
git stash pop
```

##### Reset to the last commit

- You may want to revert back to the very last commit, discarding every modification from then. This could be because you were playing around with the code or doing some minor experiments. In either cases, you can do this by, 

```{code-block}
git reset --hard HEAD 
```

- Otherwise, to just unstage the files which were staged by `git add`,

```{code-block}
git reset
```

- Refer this [stackoverflow QA](https://stackoverflow.com/questions/14075581/git-undo-all-uncommitted-or-unsaved-changes) for more details. 

### DVC

TODO