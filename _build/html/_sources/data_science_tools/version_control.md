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
