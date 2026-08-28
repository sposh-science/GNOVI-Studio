# GNOVI Studio — 2026 repository migration

This repository was rebuilt in 2026. Its `main` branch reproduces the
development history of the predecessor GNOVI Studio repository, with the
GitHub pull-request sequence reconstructed one-to-one.

## What was preserved unchanged

* All pre-migration history up to and including commit `c2fd15b`
  ("Merge pull request #9 from wavicles/chore/project-guide-neutrality")
  is imported byte-for-byte, with original commit SHAs, authors and dates.

  Note: merge commits inside that imported history refer to pull-request
  numbers (`#2`–`#9`) from an earlier repository. Those numbers are not
  related to this repository's pull requests.

## What was reconstructed

* Original pull requests #1–#25 were recreated here as pull requests #1–#25,
  in the same order. Each reconstructed PR replays the code of the original
  PR; its feature commits keep their original author dates. Merge commits
  and the committer timestamps of replayed commits carry the real migration
  date — no historical timestamp was fabricated.

* Each reconstructed PR description records the original PR number and its
  original merge commit.

## v0.9.0

The original v0.9.0 Beta was released on 2026-08-23 from the predecessor
repository (its original PR #5). In this repository, `v0.9.0` is a new
annotated tag on the reconstructed PR #5 merge, created during the
migration with a current tagger timestamp. Its source tree matches the
original v0.9.0 release on every project path.

## Authorship integrity

`githooks/` and `.github/workflows/authorship-integrity.yml` implement an
authorized-contributor allowlist: only identities listed in
`githooks/authorized-contributors` (plus the platform merge identity) may
appear as a commit Author, Committer or `Co-authored-by` trailer. The
check inspects commit metadata only. These files were added during the
migration and are the only files in this repository that did not exist in
the pre-migration history.
