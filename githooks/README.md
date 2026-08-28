# Git hooks — authorized-contributor policy

These hooks enforce Git authorship integrity: every commit Author, every
`Co-authored-by` trailer identity, and every non-platform Committer must be
listed in `authorized-contributors`.

Enable them in your clone:

    git config core.hooksPath githooks

`authorship-check.sh` reads commit metadata only (author/committer headers
and the trailer block). It does not read file contents and matches no
tool, vendor or model names. The same check runs in CI
(`.github/workflows/authorship-integrity.yml`).

Introduced during the 2026 repository migration (see `../MIGRATION.md`).
