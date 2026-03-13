# PyPI Trusted Publishing from GitHub Actions

This project uses OpenID Connect (OIDC) trusted publishing to release to PyPI without storing PyPI API tokens in GitHub secrets.

## What is configured in this repository

- Workflow file: `.github/workflows/publish-pypi.yml`
- Trigger: Git tag pushes matching `v*` (for example `v0.0.2`)
- Build job: creates `sdist` + `wheel` and runs `twine check`
- Publish job: uses `pypa/gh-action-pypi-publish` with `id-token: write`

## One-time setup in PyPI

1. Create the `umimic` project on PyPI if it does not exist yet.
2. In PyPI project settings, open `Publishing`.
3. Add a `Trusted Publisher` with:
   - Owner: your GitHub org/user
   - Repository: `python-release-readiness` (or the final repo name)
   - Workflow: `publish-pypi.yml`
   - Environment: `pypi`
4. Save the trusted publisher.

Use exact repository/workflow/environment names to match GitHub Actions claims.

## One-time setup in GitHub

1. In GitHub repository settings, create an environment named `pypi`.
2. Optionally require manual approvals for the `pypi` environment.
3. Ensure workflow permissions are not restricted from OIDC token issuance.

No PyPI token secrets are required with trusted publishing.

## Release process

1. Ensure package version is updated in:
   - `pyproject.toml`
   - `umimic/__init__.py`
2. Run release checks locally:

```bash
pytest -q
python -m build
python -m twine check dist/*
```

3. Commit and push changes.
4. Create and push a version tag:

```bash
git tag v0.0.2
git push origin v0.0.2
```

5. Watch `Publish to PyPI` workflow in GitHub Actions.
6. Verify the new release appears on PyPI.

## Troubleshooting

### `invalid-publisher` or `unauthorized` during publish

The trusted publisher tuple does not match. Re-check:

- GitHub owner
- repository name
- workflow filename
- environment name

### Workflow builds artifacts but publish is skipped/blocked

Check `pypi` environment protection rules in GitHub and approve if required.

### PyPI rejects upload due to existing version

PyPI versions are immutable. Increment version and push a new tag.
