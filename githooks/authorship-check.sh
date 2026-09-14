#!/usr/bin/env bash
# Git Authorship Review.
#
# For every commit in a range, reports the Author, Committer, and any
# attribution trailers (Co-authored-by and similar "...-by" trailers),
# classifies each identity, and fails ONLY on genuinely prohibited
# attribution -- never merely because a contributor is unrecognized.
# GNOVI Studio is open source: external human contributors keep their own
# Git authorship and are never required to be pre-approved.
#
# Classification, per identity (Author / Committer / trailer value):
#   MAINTAINER            -> exact match in maintainer-identities
#   MAINTAINER_MISMATCH   -> the *name* matches a maintainer identity's
#                            display name, but the full "Name <email>"
#                            does not -- claims to be the maintainer under
#                            an identity we don't recognize (hard failure)
#   GITHUB_PLATFORM       -> the hosting platform's own merge-commit
#                            identity (GitHub <noreply@github.com>)
#   PROHIBITED             -> exact match in prohibited-attribution
#                            (hard failure)
#   EXTERNAL               -> anything else: an ordinary, unrecognized
#                            human-looking identity. Passes; surfaced for
#                            maintainer review.
#
# A commit fails only if any of its identities classify as
# MAINTAINER_MISMATCH or PROHIBITED.
#
# This inspects commit metadata only -- the author/committer headers and
# the message trailer block (via `git interpret-trailers --parse`). It
# does not read file contents or diffs, and prose in the commit message
# body is never matched against anything.
#
# This is attribution hygiene and a review aid, NOT identity verification:
# Git author/committer/trailer fields are self-reported and can be set to
# anything by whoever makes the commit. It cannot prove a human wrote the
# change; it can only flag attribution that explicitly claims to be an
# AI/tool/service, or that claims the maintainer's identity incorrectly.
#
# In GitHub Actions (GITHUB_ACTIONS=true) this also writes a Markdown
# "Authorship Review" table to $GITHUB_STEP_SUMMARY and emits
# ::notice::/::error:: annotations. Outside CI it just prints findings.
#
#   usage: authorship-check.sh <base>..<head>
#          authorship-check.sh <base> <head>
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MAINTAINER_FILE="${AUTHORSHIP_MAINTAINER_FILE:-$HERE/maintainer-identities}"
PROHIBITED_FILE="${AUTHORSHIP_PROHIBITED_FILE:-$HERE/prohibited-attribution}"
IS_CI="${GITHUB_ACTIONS:-false}"

# The one non-human identity that legitimately signs commits/merges made
# through the GitHub web UI or API -- never valid as a human Author.
GITHUB_PLATFORM_IDENTITY="GitHub <noreply@github.com>"

case $# in
  1) RANGE="$1" ;;
  2) RANGE="$1..$2" ;;
  *) echo "usage: authorship-check.sh <base>..<head> | <base> <head>" >&2; exit 2 ;;
esac

[ -r "$MAINTAINER_FILE" ] || { echo "authorship: cannot read maintainer identity file: $MAINTAINER_FILE" >&2; exit 2; }
[ -r "$PROHIBITED_FILE" ] || { echo "authorship: cannot read prohibited-attribution file: $PROHIBITED_FILE" >&2; exit 2; }

mapfile -t MAINTAINERS < <(grep -vE '^[[:space:]]*(#|$)' "$MAINTAINER_FILE" | sed 's/[[:space:]]*$//')
mapfile -t PROHIBITED_EXACT < <(grep -vE '^[[:space:]]*(#|$)' "$PROHIBITED_FILE" | grep '<' | sed 's/[[:space:]]*$//')
mapfile -t PROHIBITED_NAMES < <(grep -vE '^[[:space:]]*(#|$)' "$PROHIBITED_FILE" | grep -v '<' | sed 's/[[:space:]]*$//')

declare -A MAINTAINER_NAME_TO_IDENTITY=()
for id in "${MAINTAINERS[@]}"; do
  MAINTAINER_NAME_TO_IDENTITY["${id%% <*}"]="$id"
done

in_list() { local n="$1"; shift; local x; for x in "$@"; do [ "$x" = "$n" ] && return 0; done; return 1; }

is_prohibited_name() {
  local name lname="$1" x lx
  lname="$(printf '%s' "$lname" | tr '[:upper:]' '[:lower:]')"
  for x in "${PROHIBITED_NAMES[@]}"; do
    lx="$(printf '%s' "$x" | tr '[:upper:]' '[:lower:]')"
    [ "$lx" = "$lname" ] && return 0
  done
  return 1
}

# classify_identity <"Name <email>"> -> echoes one classification word
classify_identity() {
  local id="$1" name="${1%% <*}"
  if in_list "$id" "${PROHIBITED_EXACT[@]}" || is_prohibited_name "$name"; then
    echo "PROHIBITED"; return
  fi
  if [ "$id" = "$GITHUB_PLATFORM_IDENTITY" ]; then
    echo "GITHUB_PLATFORM"; return
  fi
  if in_list "$id" "${MAINTAINERS[@]}"; then
    echo "MAINTAINER"; return
  fi
  if [ -n "${MAINTAINER_NAME_TO_IDENTITY[$name]+x}" ]; then
    echo "MAINTAINER_MISMATCH"; return
  fi
  echo "EXTERNAL"
}

esc_md() { printf '%s' "$1" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g'; }

revs=$(git rev-list --reverse "$RANGE") || { echo "authorship: bad range: $RANGE" >&2; exit 2; }

rc=0
commit_count=0
external_count=0
maintainer_count=0
declare -a SUMMARY_ROWS=()
declare -a EXTERNAL_SHAS=()

for sha in $revs; do
  commit_count=$((commit_count + 1))
  short="${sha:0:7}"
  author="$(git log -1 --format='%an <%ae>' "$sha")"
  committer="$(git log -1 --format='%cn <%ce>' "$sha")"

  author_cls="$(classify_identity "$author")"
  committer_cls="$(classify_identity "$committer")"

  declare -a co_author_display=()
  declare -a trailer_problems=()
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    key="${line%%:*}"
    val="$(printf '%s' "${line#*:}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
    [ -z "$val" ] && continue

    val_cls="$(classify_identity "$val")"
    if [ "$val_cls" = "PROHIBITED" ] || [ "$val_cls" = "MAINTAINER_MISMATCH" ]; then
      trailer_problems+=("$key: $val ($val_cls)")
    fi

    shopt -s nocasematch
    if [[ "$key" == *-by ]]; then
      co_author_display+=("$key: $val")
    fi
    shopt -u nocasematch
  done < <(git log -1 --format='%B' "$sha" | git interpret-trailers --parse 2>/dev/null)

  commit_failed=0
  if [ "$author_cls" = "PROHIBITED" ]; then
    echo "REJECT $sha  prohibited Author: $author" >&2
    commit_failed=1
  elif [ "$author_cls" = "MAINTAINER_MISMATCH" ]; then
    echo "REJECT $sha  Author claims maintainer identity '${author%% <*}' but does not match an approved maintainer identity: $author" >&2
    commit_failed=1
  fi
  if [ "$committer_cls" = "PROHIBITED" ]; then
    echo "REJECT $sha  prohibited Committer: $committer" >&2
    commit_failed=1
  elif [ "$committer_cls" = "MAINTAINER_MISMATCH" ]; then
    echo "REJECT $sha  Committer claims maintainer identity '${committer%% <*}' but does not match an approved maintainer identity: $committer" >&2
    commit_failed=1
  fi
  for problem in "${trailer_problems[@]+"${trailer_problems[@]}"}"; do
    echo "REJECT $sha  $problem" >&2
    commit_failed=1
  done

  if [ "$commit_failed" -eq 1 ]; then
    rc=1
    review="🔴 Prohibited or mismatched attribution -- see REJECT lines"
    [ "$IS_CI" = "true" ] && echo "::error title=Authorship Review::$short: prohibited or mismatched attribution (see job log)."
  elif [ "$author_cls" = "MAINTAINER" ]; then
    maintainer_count=$((maintainer_count + 1))
    review="🟢 Known maintainer"
  elif [ "$author_cls" = "GITHUB_PLATFORM" ]; then
    review="🟢 GitHub platform commit"
  else
    external_count=$((external_count + 1))
    EXTERNAL_SHAS+=("$short")
    review="🟡 External contributor -- verify before merge"
  fi

  co_authors_display="None"
  if [ "${#co_author_display[@]}" -gt 0 ]; then
    co_authors_display="$(printf '%s<br>' "${co_author_display[@]}")"
    co_authors_display="${co_authors_display%<br>}"
  fi

  SUMMARY_ROWS+=("| $short | $(esc_md "$author") | $(esc_md "$committer") | $(esc_md "$co_authors_display") | $review |")

  unset co_author_display trailer_problems
done

if [ "$rc" -ne 0 ]; then
  echo "" >&2
  echo "authorship: prohibited or mismatched attribution found (see REJECT lines above)." >&2
  echo "AI/tool/model/assistant identities may never appear as a commit Author," >&2
  echo "Committer, or attribution trailer. See githooks/prohibited-attribution." >&2
  echo "A commit claiming to be the maintainer must use an identity listed in" >&2
  echo "githooks/maintainer-identities." >&2
elif [ "$external_count" -gt 0 ]; then
  echo "authorship: $external_count external-contributor commit(s) -- review recommended, not a failure." >&2
fi

if [ "$IS_CI" = "true" ] && [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    echo "## Authorship Review"
    echo ""
    echo "Commits inspected: $commit_count"
    echo ""
    echo "| Commit | Author | Committer | Co-authors | Review |"
    echo "|---|---|---|---|---|"
    for row in "${SUMMARY_ROWS[@]}"; do echo "$row"; done
    echo ""
    if [ "$rc" -ne 0 ]; then
      echo "**Result:** ❌ Prohibited or mismatched attribution detected."
    else
      echo "**Result:** ✅ No prohibited attribution detected."
    fi
    if [ "$maintainer_count" -gt 0 ]; then
      echo ""
      echo "Known maintainer: $maintainer_count"
    fi
    if [ "$external_count" -gt 0 ]; then
      echo ""
      echo "External contributor: $external_count -- verify attribution before merging."
    fi
  } >> "$GITHUB_STEP_SUMMARY"

  if [ "$external_count" -gt 0 ]; then
    echo "::notice::External contributor detected. Review Git attribution before merging."
  fi
fi

exit "$rc"
