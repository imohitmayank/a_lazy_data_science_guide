Github Good Practices
=============

## Introduction

[GitHub](https://github.com/) is a powerful tool for version control and collaboration. In this guide, we will cover essential practices for using GitHub effectively. Whether you are a beginner or an experienced developer, adhering to these practices will help you and your team work more efficiently and maintain a high standard of code quality.

## Practices

Let's go through individual steps in the lifecycle of using Github to discuss good practices,

### Repositories

- **Creating Repo:** Create a new repository when starting a new project, developing a standalone feature, or creating a reusable component. That said, we should be careful about not creating separate projects if the individual modules are supposed to be used together. For example, if we are creating a dashboard *(say, using React)* that requires specific endpoints *(say, using Python)*, we can combine the into a single repo and run both applications.
- **Project Owner**: Each repo should have at least one Spock or owner who is responsible for new implementation or reviewing any changes made by others in form of PR. This helps with concentrating ownership and responsibility. 
- **Repo Structure:** The organization of files and directories within a repository is crucial for maintainability and efficiency. It largely depends on the specific language or set of languages used in the project. A well-structured repository should have a logical hierarchy, clear separation of concerns, and follow established conventions for the chosen technology stack. Refer [Python Good Practices](./python_good_practices.md#project-structuring) for more details.
- **Use Descriptive Names:** Choose repository names that clearly indicate the project's purpose or content. A good practices is to name the repo with following nomenclature `{team}.{projectname}.{service}`. As an example, if AI team is  developing a Content Classifier that will be exposed as APIs, then a suitable name would be `ai.contentclassifer.api`
- **Include a README:** Always create a comprehensive README file that explains the project's purpose, setup instructions, and usage guidelines.

### Branches

- **Use Feature Branches:** Create separate branches for each new feature or bug fix. This isolates changes and makes it easier to merge them into the main branch when they're ready. Properly naming a branch is crucial for effective and collaborative execution. Some examples are,
  - **Feature branch:** `feature/new-user-registration`
  - **Bugfix branch:** `bugfix/login-error`
  - **Hotfix branch:** `hotfix/critical-security-issue`
  - **Release branch:** `release/v1.2.0`
  - **Individual branch**: `mohit/10-10-2024-updates`

- **Keep Branches Up-to-Date:** Regularly merge or rebase your feature branches with the `dev` or `main` or `master` branch *(whichever branch you intend to merge to later)* to avoid merge conflicts and ensure your code is always compatible with the latest changes.

!!! Warning
    Do not push changes directly to main branches. Make sure to create a feature branch, make changes there, test them, and then raise a PR to merge the changes into the main branch. More on this below. 

### Committing Changes

- **Commit Frequently:** Make small, focused commits that introduce a single change or feature. This makes it easier to review and revert changes if necessary.
- **Write Clear Commit Messages:** Use descriptive commit messages that accurately convey the purpose of the change. Some good practices are,
  - **Use the imperative mood:** Start your commit message with a verb (e.g., "Add," "Fix," "Improve").
  - **Refer to issues or pull requests:** If your commit is related to a specific issue or pull request, include a reference to it in the commit message. Ex: `Fixed #OB-101`
  - **Keep it short:** Aim for a commit message that is no more than 50 characters long in general.

### Pull Requests

- **Pre-PR:** Before even raising a PR, make sure you
  - **Thoroughly Test Changes:** Before creating a pull request, ensure your changes work as expected by running unit tests and/or manual testing. It is also important to perform a thorough test of other modules that might have been impacted due to the changes and perform a random check on other modules to make sure there is no sporadic impact on the overall project.
  - **Fix Issues Locally:** Address any bugs or issues you discover during testing before submitting your pull request.
    
!!! Warning
    DO NOT raise a PR if the work is incomplete, or has not been tested. Remember, by raising a PR you should mean to convey that “I have completed the development, and have performed proper tests”. Ask yourself this question, before raising a PR.  
    
- **Draft PR:** You can also raise an Draft PR incase you need an early review of the completed modules while you are still working on dependent modules. This helps to expedite the review process, and reduce reviewer’s load by providing them with sufficient time.
- **Creating a PR:**
  - **Create Descriptive Pull Requests:** Use clear and concise titles and descriptions that explain the purpose of the changes. Make sure the title highlight the most important aspects of the PR and the description contains individual changelogs. One example could be,
        
    ```markdown
    ## Title
    Implement User Authentication Feature
    
    ## Description 
    **Changelog:**
    - Added login and registration forms
    - Implemented backend API endpoints for authentication
    - Set up JWT token generation and validation
    ```
    
  - **Add reviewers:** When creating a pull request, assign appropriate team members to review your code. Choose reviewers who are familiar with the project and can provide valuable feedback.

- **Post-PR:**
  - **Review Code Carefully:** Reviewer should thoroughly go through the changelog in code and suggest improvements if any by adding comments. Do not approve the PR unless you are satisfied with the code’s quality, impact and implementation.
  - **Address Feedback Promptly:** Conversations to comments and suggestions from reviewers should be addressed in a timely manner.

## Conclusion

By following these GitHub best practices, you can ensure a more organized, efficient, and collaborative workflow. These practices help maintain high code quality, reduce the likelihood of errors, and streamline the development process. Remember, consistency is key, and adopting these practices as a team will lead to better project outcomes and a more productive development environment.

Happy coding!