Version Control
========================
------------

## GIT

- GIT is the defacto version control for code base, with popular free and services like GitHub, Gitlab and BitBucket.

### Modify config to add email and name

```{code-block} bash
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

### Ignore files/folders

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

### Untrack file/folder and delete them from GIT

- To untrack the files or folders, we can create `.gitignore` file and add respective info.
- To delete the files or folders form GIT (and not from local system), we can delete them from the cache as suggested [here](https://stackoverflow.com/questions/1143796/remove-a-file-from-a-git-repository-without-deleting-it-from-the-local-filesyste),

```{code-block}
# for a single file:
git rm --cached mylogfile.log

# for a single directory:
git rm --cached -r mydirectory
```
