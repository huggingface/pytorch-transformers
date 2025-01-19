import os
import logging
from datetime import datetime as dt
from typing import List

import github
from github import Github, GithubException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LABELS_TO_EXEMPT = [
    "good first issue",
    "good second issue",
    "good difficult issue",
    "feature request",
    "new model",
    "wip",
]

STALE_COMMENT = (
    "This issue has been automatically marked as stale because it has not had "
    "recent activity. If you think this still needs to be addressed "
    "please comment on this thread.\n\nPlease note that issues that do not follow the "
    "[contributing guidelines](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md) "
    "are likely to be ignored."
)


def close_stale_issues(repo):
    """Close stale issues based on the criteria defined."""
    open_issues = repo.get_issues(state="open")

    for i, issue in enumerate(open_issues):
        logger.info(f"Processing issue #{issue.number}: {issue.title}")
        comments = sorted(issue.get_comments(), key=lambda c: c.created_at, reverse=True)
        last_comment = comments[0] if comments else None

        if should_close_issue(issue, last_comment):
            close_issue(issue)
        elif should_mark_stale(issue):
            mark_issue_stale(issue)


def should_close_issue(issue, last_comment) -> bool:
    """Determine if an issue should be closed."""
    return (
        last_comment is not None and last_comment.user.login == "github-actions[bot]"
        and (dt.utcnow() - issue.updated_at.replace(tzinfo=None)).days > 7
        and (dt.utcnow() - issue.created_at.replace(tzinfo=None)).days >= 30
        and not any(label.name.lower() in LABELS_TO_EXEMPT for label in issue.get_labels())
    )


def close_issue(issue):
    """Close the given issue."""
    try:
        issue.edit(state="closed")
        logger.info(f"Closed issue #{issue.number}")
    except GithubException as e:
        logger.error(f"Couldn't close issue #{issue.number}: {repr(e)}")


def should_mark_stale(issue) -> bool:
    """Determine if an issue should be marked as stale."""
    return (
        (dt.utcnow() - issue.updated_at.replace(tzinfo=None)).days > 23
        and (dt.utcnow() - issue.created_at.replace(tzinfo=None)).days >= 30
        and not any(label.name.lower() in LABELS_TO_EXEMPT for label in issue.get_labels())
    )


def mark_issue_stale(issue):
    """Mark the given issue as stale by creating a comment."""
    try:
        issue.create_comment(STALE_COMMENT)
        logger.info(f"Marked issue #{issue.number} as stale.")
    except GithubException as e:
        logger.error(f"Couldn't create comment for issue #{issue.number}: {repr(e)}")


def main():
    """Main function to close stale issues."""
    g = Github(os.environ["GITHUB_TOKEN"])
    repo = g.get_repo("huggingface/transformers")
    close_stale_issues(repo)


if __name__ == "__main__":
    main()
