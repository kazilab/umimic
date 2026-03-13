# Read the Docs Implementation from GitHub

This guide documents how to host U-MIMIC documentation on Read the Docs (RTD) with GitHub as the source of truth.

## Architecture

1. Code and docs live in GitHub.
2. RTD imports the GitHub repository.
3. Pushes and pull requests trigger RTD builds.
4. RTD serves versioned docs and PR previews.

## Repository files required

The repository now includes:

- `.readthedocs.yaml`: RTD build configuration
- `docs/conf.py`: Sphinx config
- `docs/index.md`: docs landing page
- this guide (`docs/readthedocs-github.md`)

## 1. Import the repository into Read the Docs

1. Sign in to RTD with your GitHub account.
2. Click `Import a Project`.
3. Select the U-MIMIC GitHub repository.
4. Confirm the default branch (usually `main`).

RTD will detect `.readthedocs.yaml` at repository root and use it for builds.

## 2. Connect GitHub webhooks

After importing:

1. Open RTD project `Admin` -> `Integrations`.
2. Ensure GitHub integration webhook is active.
3. In GitHub, check `Settings` -> `Webhooks` for an RTD webhook entry.

This ensures docs rebuild on push and pull request events.

## 3. Build configuration details

RTD uses `.readthedocs.yaml`:

- Python: `3.11`
- OS image: Ubuntu 24.04
- Builder: Sphinx with config at `docs/conf.py`
- Install command: `pip install .[docs]`

`.[docs]` comes from `pyproject.toml` and currently installs:

- `sphinx`
- `myst-parser`

## 4. Local validation before pushing

Run docs build locally before opening a PR:

```bash
python -m pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

If build succeeds, open `docs/_build/html/index.html` to preview.

## 5. Configure versions and preview behavior in RTD

In RTD project settings:

1. `Versions`: enable at least `latest` and `stable`.
2. `Automation Rules`: enable pull request builds.
3. Optionally hide inactive branches to reduce noise.

Recommended policy:

- `stable`: latest tagged release branch/tag
- `latest`: default branch (`main`)
- PR builds: enabled for contributor review

## 6. Add docs status to GitHub (optional)

You can add the RTD badge to `README.md` and enable required checks in branch protection:

- RTD build status
- repo test workflow(s)

This prevents merges that break documentation.

## 7. Troubleshooting

### Build fails with missing dependencies

Cause: docs dependency not included in `pyproject.toml` extras.  
Fix: add package to `[project.optional-dependencies].docs` and retry build.

### RTD does not rebuild on push

Cause: webhook missing or disabled.  
Fix: re-sync integration from RTD `Admin -> Integrations`.

### Docs page renders empty navigation

Cause: missing `toctree` entries in `docs/index.md`.  
Fix: include each documentation page in the `toctree`.

## Operational checklist

Before each release:

1. Build docs locally with Sphinx.
2. Push docs changes to GitHub.
3. Confirm RTD build is green for `latest`.
4. Create release tag.
5. Confirm `stable` version is updated in RTD.

## Acknowledgements

Parts of this documentation were created with assistance from ChatGPT Codex and Claude Code.
