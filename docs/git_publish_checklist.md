# Git publication checklist

## Required before first push

- [ ] Decide whether the repository is private or public.
- [ ] Replace `LICENSE_PENDING.md` with an approved license or keep the repository private.
- [ ] Add real author/organization metadata only after approval.
- [ ] Confirm that model, dataset, and vendor-runtime licenses permit any external downloads you publish.
- [ ] Run `python scripts/repo_check.py`.
- [ ] Run `python -m unittest discover -s tests -v` (and optionally `python -m pytest -q` after installing the development requirements).
- [ ] Inspect `git status --short` for `.env`, credentials, hostnames, model files, caches, data, logs, and private paths.
- [ ] Confirm no file exceeds the hosting provider's normal Git limit.
- [ ] Publish external artifact SHA256 values and access instructions without embedding signed/private URLs.

## Suggested commands

```bash
git init
git add .
git status --short
python scripts/repo_check.py
git commit -m "Initial auditable K100 deployment release"
git branch -M main
git remote add origin <YOUR-REPOSITORY-URL>
git push -u origin main
```

Do not paste a personal access token into a shell script or tracked remote URL. Use the credential manager or SSH agent provided by the Git hosting service.

## Large artifacts

Prefer release assets or object storage for immutable checkpoints/ONNX/MXR bundles. Git LFS is acceptable only if storage quotas and redistribution permissions are understood. Always publish a detached size/SHA256 manifest.
