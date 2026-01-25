# Upload kit for `open-bathy-workflows`

This zip contains everything you need to create and push your new repo **programmatically**.

## Prerequisites

- `git` installed
- `unzip` installed
- GitHub repo already created on the website:
  - https://github.com/camante/open-bathy-workflows
- Authentication set up:
  - Recommended: SSH (git@github.com:camante/open-bathy-workflows.git)
  - Or HTTPS with a Personal Access Token (PAT)

## Quick start (SSH recommended)

1) Download/unzip this kit
2) In a terminal, `cd` into the unzipped folder (the one containing `bootstrap_repo.sh`)
3) Run:

```bash
export REMOTE_URL="git@github.com:camante/open-bathy-workflows.git"
export GIT_USER_NAME="Your Name"
export GIT_USER_EMAIL="you@example.com"
bash bootstrap_repo.sh
```

## Quick start (HTTPS)

```bash
export REMOTE_URL="https://github.com/camante/open-bathy-workflows.git"
export GIT_USER_NAME="Your Name"
export GIT_USER_EMAIL="you@example.com"
bash bootstrap_repo.sh
```

If you use HTTPS, git will prompt you for credentials. Use:
- username: your GitHub username
- password: your GitHub **Personal Access Token (PAT)** (not your GitHub password)

## What the script does

- Creates `open-bathy-workflows/`
- Copies scaffold repo files (.gitignore, README, LICENSE, env)
- Unzips the workflow snapshot into `open-bathy-workflows/workflow/`
- Initializes git, commits, tags `v0.1.0`
- Adds remote and pushes `main` + tags

## Notes

- The workflow snapshot zip is included at:
  - `workflow_source/v0_8_0_aniso_fix.zip`
- If you want to swap in a newer workflow zip, replace that file and run:
  - `export WORKFLOW_ZIP=workflow_source/<your_zip_name>.zip`
