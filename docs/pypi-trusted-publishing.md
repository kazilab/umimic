# PyPI Trusted Publishing from GitHub Actions

OpenID Connect (OIDC) trusted publishing can release to PyPI without storing
PyPI API tokens in GitHub secrets. Publishing requires a configured GitHub
workflow and a matching trusted publisher in the PyPI account. The workflow is
in the repository (`.github/workflows/publish-pypi.yml`); the one-time PyPI and
GitHub setup below still has to be done in those accounts.

## Required workflow configuration

- Workflow file: `.github/workflows/publish-pypi.yml`
- Trigger: Git tag pushes matching `v*` (for example `v0.0.4`)
- Build job: creates `sdist` + `wheel` and runs `twine check`
- Publish job: uses `pypa/gh-action-pypi-publish` with `id-token: write`

## One-time setup in PyPI

1. Create the `umimic` project on PyPI if it does not exist yet.
2. In PyPI project settings, open `Publishing`.
3. Add a `Trusted Publisher` with:
   - Owner: your GitHub org/user
   - Repository: `umimic` (or the actual repository name)
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

1. Update `__version__` in `umimic/__init__.py`. The build configuration reads
   that value, so package metadata and `umimic --version` use the same version.
2. Run release checks locally:

```bash
pytest -q
python -m build
python -m twine check dist/*
```

3. Commit and push changes.
4. Create and push a version tag:

```bash
git tag v0.0.4
git push origin v0.0.4
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
