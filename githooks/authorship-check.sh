#!/usr/bin/env bash
# Git authorship integrity check.
#
# Verifies that every commit in a range carries only authorized identities
# in its metadata:
#   - Author            -> must be in githooks/authorized-contributors
#   - Co-authored-by:    -> every trailer identity must be in that file
#   - Committer          -> must be in that file, OR an authorized platform
#                           merge identity (a hosting account that signs
#                           merges created through the web UI / API)
#
# It inspects commit metadata only: the author/committer headers and the
# message trailer block (via `git interpret-trailers --parse`). It does
# NOT read file contents, diffs, or message prose, and it matches no
# tool, vendor, model or company names.
#
#   usage: authorship-check.sh <base>..<head>
#          authorship-check.sh <base> <head>
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ALLOW_FILE="${AUTHORSHIP_ALLOWLIST:-$HERE/authorized-contributors}"

# Valid only as Committer (never as Author or Co-author): platform accounts
# that sign merge commits created through the hosting UI / API.
AUTHORIZED_MERGE_COMMITTERS=(
  "GitHub <noreply@github.com>"
)

case $# in
  1) RANGE="$1" ;;
  2) RANGE="$1..$2" ;;
  *) echo "usage: authorship-check.sh <base>..<head> | <base> <head>" >&2; exit 2 ;;
esac

[ -r "$ALLOW_FILE" ] || { echo "authorship: cannot read allowlist: $ALLOW_FILE" >&2; exit 2; }
mapfile -t ALLOW < <(grep -vE '^[[:space:]]*(#|$)' "$ALLOW_FILE" | sed 's/[[:space:]]*$//')

in_list() { local n="$1"; shift; local x; for x in "$@"; do [ "$x" = "$n" ] && return 0; done; return 1; }

revs=$(git rev-list "$RANGE") || { echo "authorship: bad range: $RANGE" >&2; exit 2; }

rc=0
for sha in $revs; do
  author="$(git log -1 --format='%an <%ae>' "$sha")"
  committer="$(git log -1 --format='%cn <%ce>' "$sha")"

  in_list "$author" "${ALLOW[@]}" || { echo "REJECT $sha  unauthorized Author: $author"; rc=1; }

  if ! in_list "$committer" "${ALLOW[@]}"; then
    in_list "$committer" "${AUTHORIZED_MERGE_COMMITTERS[@]}" \
      || { echo "REJECT $sha  unauthorized Committer: $committer"; rc=1; }
  fi

  while IFS= read -r line; do
    key="${line%%:*}"
    shopt -s nocasematch
    if [[ "$key" == "co-authored-by" ]]; then
      shopt -u nocasematch
      id="$(printf '%s' "${line#*:}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
      in_list "$id" "${ALLOW[@]}" || { echo "REJECT $sha  unauthorized Co-authored-by: $id"; rc=1; }
    else
      shopt -u nocasematch
    fi
  done < <(git log -1 --format='%B' "$sha" | git interpret-trailers --parse 2>/dev/null)
done

if [ "$rc" -ne 0 ]; then
  echo "" >&2
  echo "authorship: unauthorized identity in commit metadata (see REJECT lines)." >&2
  echo "If a listed identity is a genuine collaborator, add it to" >&2
  echo "githooks/authorized-contributors in a reviewed commit." >&2
fi
exit "$rc"
