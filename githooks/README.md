# Git hooks — Authorship Review

These hooks run `authorship-check.sh`, GNOVI Studio's Authorship Review:
for every commit in range it reports the Author, Committer, and any
attribution trailers (`Co-authored-by` and similar), and fails only for
attribution that is genuinely prohibited:

- an explicit AI/tool/model/assistant identity (see
  `prohibited-attribution`) as Author, Committer, or a trailer;
- a commit claiming the maintainer's name (see `maintainer-identities`)
  under an identity that doesn't match.

Ordinary human contributors -- known or unknown -- are never rejected for
being unrecognized. External contributors keep their own Git authorship
and do not need to be pre-approved.

Enable the hooks in your clone:

    git config core.hooksPath githooks

`authorship-check.sh` reads commit metadata only (author/committer
headers and the trailer block). It does not read file contents, and
prose in the commit message is never matched against anything. This is
attribution hygiene and a review aid, not identity verification -- Git
metadata is self-reported and cannot prove who actually wrote a change.

The same check runs in CI (`.github/workflows/authorship-integrity.yml`),
where it also writes a Markdown "Authorship Review" table to the GitHub
Actions Job Summary and annotates external contributions for maintainer
review.

Tests: `githooks/test-authorship-check.sh`.

Introduced during the 2026 repository migration (see `../MIGRATION.md`);
redesigned from an allowlist to a review model after PR #38 showed the
allowlist rejected every new external contributor by default.
