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

- In simple words, GIT is a system designed to track changes in your file. True story, it was developed by none other but the creator of Linux, yes, Linus Torvalds in 2005. The story goes something like this -- while he was developing the linux kernel along with other kernel developers, he found it troublesome to maintain, track and handle conflicting (overlapping) pieces of codes. So he ended up coding the GIT system as a side project, just to help him and his fellow developers maintain linux kernel in a more efficient manner! Now, isn't that a cod side project 😎. You can read more about GIT and the history [here](https://en.wikipedia.org/wiki/Git).

- Some of the popular and free services are GitHub, Gitlab and BitBucket. While some of them may have better UI or add-on functionalities, the basic system used by all of them is the same. Hence if you learn the basic command once, you can use GIT no matter the platform. And that is why we will limit ourselves to learn GIT the old school way (via GIT bash command), and leave the fancy UI or tool based operation as an exploration activity for the interested audience. 

- Before we go into the command and syntax, we need to be clear about certain topics, to better appreciate GIT. These are, 
  
    - **From where can we get Git?** Git is free and available [here](https://git-scm.com/). Download the latest stable version as per your OS. 
    - **How do we use Git?** After downloading Git *(and specifically in Windows)*, from any directory in file explorer, on right click, you should get the option to open either "Git bash" or "Git GUI". For this article, we will use Git bash, as that's how real developers roll 😤 *(jk)*
    - **What is Git Bash?** It is something similar to command prompt in windows or terminal in linux, but something specifically designed for Git. To perform any Git related operation, we will use the Git bash.
    - **What is local vs remote server?** Local instance is a git repository present in your local computer, and on the other hand remote instance is connecting the local instance to an rmote server. This is done so that there is a centralised repository from where eeryone in the team can pull or push latest code.
    - **What are branches in Git?** Branches are like parallel universes in Git. We can spawn off new branches anytime and from any other branch. By doing so we create a fork within that branch, where the developer can do whatever they want to do. Finally, after relevant commits the forked branch is merged back to their original branch using merge request.
    - **What is a merge request and merge conflict in Git?** Merge request is the request to merge two branches. It could happen that the same lines in the same file has been modified in both the branches, this will lead to a merge conflict. Resolving merge conflict can range from easy to highly complex based on how many files has been be affected. The idea is to select the choose the line of codes that should finally persist in the final version.
    - **What is the life cycle in Git?** Basically any file in a git repository typically goes through three states. These are, (1) working state: the initial stage, where you have created a git repository and git is just looking at the files to note what have changed. (2) staging state: the second state where you set certiain file that should be commited, and (3) commit state: where you finalize the changes made to the files and commit them to git's history. Basically, this will create a version of your code and persist it for future reference.  

- Now let's take a practical approach of learning Git. Suppose you are working on a big project with lots of modifications happening per day. Now you are looking for a tool, that will help you keep track of the changes you make by recording the line wise modification made by you in the files in your project. For this you want to explore GIT and see if it will make your life easier or not. So, let's get stated with the exploration! 😀
  
    - **Initializing the Git repository:** As we discussed, Git helps in tracking the changes made in a file. On top of it, it's easy to scale it up and track a complete folder that contains hundreds of files! For reference, suppose you have already created a project (follow for more details on how to structure a python project), and want to track the changes in any of the files present there. To do so, just go to the main directory of the project and initialize git by using command `git init` . This will mark that directory as a git repository. Next, if you run `git status`, it will show you an overview of the project and all of files. Note, by default Git keeps a look out at all of the files within the directory and show when any of the files have changed.

    - **Tracking files:** It may so happens that you only want to really track a few files and not all. This option is available and can be done by command `git add <file_name>`, or if you want to track all of the files do `git add .` By doing this you ave moved the files into staging area. Now if you run git status again, you can see the files names are in green. This means these files are tracked!

    ```{figure} /imgs/git_init_and_track.png
    ---
    height: 400px
    ---
    Example for initializing the git repository to tracking the files.
    ```

    - **Commit:** Now, suppose we just completed one small task (adding a new function), and want to log off. It would be a good idea, if we can somehow take a snapshot of our current code and save it, so that when we come back we can easily continue. This can be done by `git commit -m "your message"` wherein you are asking git to commit all of the changes added in the staging area. This commit can be thought of as a unique snapshot of your code with the commit message as your description.

    - **Push:** Now you just remembered, your team mate who works from other side of the globe asked you to share the code once done. So before logging off, we can push or commit to remote server by `git push origin master` . Note, here `git push` signifies we want to pus the code, `origin` denotes the remote server and `master` denotes the branch of `origin` on which we want to push.


- And that's it! We have covered most of the fundamental aspects of using git. One important aspect still remains, that is we should refrain from committing directly to master branch. Instead whenever we are planning to do a modification, we should checkout to a new git branch (by using `git checkout -b <branch_name>`), do the modifications there, push that particular branch to remote and then create merge request. This is just a good practice followed when working with a team! 

#### GIT Snippets

- A consolication of some of the most helper code snippets for GIT

##### The basic git commands

- Listing down some of the most basic GIT commands, that you should definitely know about. Most of them are references from the above theory part.

```{code-block}
---
lineno-start: 1
---
# pull the latest code from "master" branch of "origin" remote server
git pull origin master
# checkout to a new branch
git checkout -b use_bert_model
# after performing some changes, add files to staging state 
git add .
# commit
git commit -m "added bert model"
# push the branch to remote
git push origin use_bert_model
```

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

## References

- [GIT](https://git-scm.com/)
- [DVC](https://dvc.org/)