# Contributing to GNOVI Studio

GNOVI Studio is an open-source scientific plotting and analysis
application. Contributions are welcome from users, scientists, educators,
developers, testers, documentation writers, and anyone else who finds it
useful — code is only one kind of contribution.

## Before contributing

For anything substantial — a new feature, a change to how something
scientific is computed, a redesign of part of the UI — opening an issue
first (or commenting on an existing one) is a good idea, so the approach
can be discussed before you invest time in it.

For small documentation fixes and clearly scoped improvements, feel free
to go straight to a pull request.

## Typical workflow

1. Fork the repository (or clone it directly if you have push access).
2. Create a focused branch for your change.
3. Make the change. Keep it scoped to one thing.
4. Test it (see below).
5. Open a pull request describing what changed and why.
6. Respond to review — a PR is usually the start of a short conversation,
   not a one-shot submission.

## Scientific and technical contributions

GNOVI Studio's fitting, plotting, and analysis code exists to produce
scientifically correct results, not just code that runs. If you're
changing a numerical method, a fitting routine, a plotting/analysis
workflow, or anything that affects computed results, please explain your
reasoning in the PR, and add tests or reference values (a closed-form
result, an independently computed check, a citation) where that makes
sense. `PROJECT_GUIDE.md` describes how the existing analysis modules are
structured and tested — it's a useful reference before making changes in
that area.

## Testing

Run the tests relevant to your change locally before opening a PR:

```bash
pip install -e ".[test,xrd]"
pytest
```

(See `README.md` for the full setup.) You don't need to chase every
possible local configuration — GitHub Actions runs the complete suite
across Debian, Ubuntu, Fedora, and Windows on every pull request, and
that's the authoritative cross-platform check. A documentation-only or
`githooks/`-only PR skips that platform matrix automatically (see
`.github/workflows/ci.yml`); everything else runs it in full.

## Pull requests

A good PR is:

- **Focused** — one change, not a bundle of unrelated ones.
- **Described** — what changed, why, and what you tested.
- **Illustrated when useful** — a before/after screenshot or a short
  description of the interaction helps a lot for GUI changes.

Small, clear PRs get reviewed faster than large ones.

## Authorship and attribution

Genuine contributors retain their own Git authorship. You don't need to
pre-register yourself with the project, ask permission, or be added to
any list before submitting a PR under your own name — contributions are
reviewed on their content and provenance, not on whether you're already
known to the project. Maintainer commits use the project's approved
maintainer identity, and that's the only identity-related rule for
project members.

The one thing we ask: don't attribute a commit's authorship or
co-authorship to an AI tool, model, or assistant, and don't alter or
rewrite another contributor's genuine authorship. `githooks/README.md`
has the details of how this is checked (it's a lightweight review aid,
not a gate you need to work around) if you're curious.

## Code and documentation style

Follow the conventions already used in the surrounding code or doc you're
touching, and keep the change as small as it can reasonably be while
still doing the job. There's no separate style guide beyond that.

## Questions

Use GitHub Issues for questions, bug reports, and feature discussion, and
Pull Requests for proposed changes.
