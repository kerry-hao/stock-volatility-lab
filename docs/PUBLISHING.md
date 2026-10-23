# Publishing to GitHub

## An empty repository

Create a GitHub repository, preferably without an initial README. Extract the
ZIP and run the following commands **inside its `optiver-volatility-lab` folder**.
Replace the final URL with the URL of the repository you actually created.

```bash
git init
git add .
git commit -m "Add volatility learning pipeline and notebooks"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/optiver-volatility-lab.git
git push -u origin main
```

Alternatively, upload the folder's contents through GitHub's web interface.
Include `.gitignore` and `.github/workflows/checks.yml` if your file picker
hides names beginning with a dot. The ZIP contains no `.git` folder.

## An existing repository

Clone it first. Copy this ZIP's contents into the desired project directory,
review any same-named files before replacing them, then commit and push from
that clone. Do not run `git init` over an existing checkout or force-push to
resolve unrelated-history errors. A project subfolder can be useful for a
larger learning portfolio; run installation and project commands from there.
The included Actions workflow assumes this project is at the repository root;
adjust its working directory if you place it in a subfolder.

## Suggested GitHub presentation

Description: `Order-book feature engineering and grouped evaluation for realized-volatility prediction.`

Topics: `python`, `machine-learning`, `quantitative-finance`, `lightgbm`,
`market-microstructure`, `volatility`, `kaggle`.

Describe the initial version as a learning pipeline with evaluation pending.
Add measured results after running it. Preserve the source attribution and
keep competition data, credentials, feature caches and model outputs out of
the initial code commit.
