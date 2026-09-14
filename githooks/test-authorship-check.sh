#!/usr/bin/env bash
# Tests for githooks/authorship-check.sh -- the Authorship Review checker.
#
# Builds throwaway synthetic Git repos (never this repository's real
# history) and drives the real, shipped githooks/maintainer-identities and
# githooks/prohibited-attribution policy files against them, so these
# tests validate the actual configuration GNOVI Studio ships, not a
# stand-in. All fixture identities are fabricated; none belong to a real
# contributor.
#
#   usage: githooks/test-authorship-check.sh
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CHECK="$HERE/authorship-check.sh"
export AUTHORSHIP_MAINTAINER_FILE="$HERE/maintainer-identities"
export AUTHORSHIP_PROHIBITED_FILE="$HERE/prohibited-attribution"

pass=0
fail=0

# --- fixture repo helpers ---------------------------------------------------

_repo=""
new_repo() {
  _repo=$(mktemp -d)
  git -C "$_repo" init -q -b main
  git -C "$_repo" config user.name "Fixture"
  git -C "$_repo" config user.email "fixture@example.invalid"
  git -C "$_repo" commit -q --allow-empty -m "root"
}

rm_repo() { [ -n "$_repo" ] && rm -rf "$_repo"; }

# commit <author-name> <author-email> <committer-name> <committer-email> <subject> [trailer...]
commit() {
  local an="$1" ae="$2" cn="$3" ce="$4" subject="$5"; shift 5
  local -a trailer_args=()
  local t
  for t in "$@"; do trailer_args+=(--trailer "$t"); done
  GIT_AUTHOR_NAME="$an" GIT_AUTHOR_EMAIL="$ae" \
  GIT_COMMITTER_NAME="$cn" GIT_COMMITTER_EMAIL="$ce" \
    git -C "$_repo" commit -q --allow-empty -m "$subject" "${trailer_args[@]}"
}

sha() { git -C "$_repo" rev-parse HEAD; }
base_sha() { git -C "$_repo" rev-parse main~0 2>/dev/null || true; }

# --- assertions --------------------------------------------------------------

assert_pass() {
  local desc="$1" base="$2" head="$3"
  local out rc
  out=$(cd "$_repo" && "$CHECK" "$base" "$head" 2>&1); rc=$?
  if [ "$rc" -eq 0 ]; then
    pass=$((pass + 1)); echo "ok    - $desc"
  else
    fail=$((fail + 1)); echo "FAIL  - $desc (expected exit 0, got $rc)"; echo "$out" | sed 's/^/        /'
  fi
}

assert_fail() {
  local desc="$1" base="$2" head="$3" expect_grep="${4:-}"
  local out rc
  out=$(cd "$_repo" && "$CHECK" "$base" "$head" 2>&1); rc=$?
  if [ "$rc" -ne 0 ]; then
    if [ -n "$expect_grep" ] && ! printf '%s' "$out" | grep -qE "$expect_grep"; then
      fail=$((fail + 1)); echo "FAIL  - $desc (rejected, but message didn't match /$expect_grep/)"; echo "$out" | sed 's/^/        /'
    else
      pass=$((pass + 1)); echo "ok    - $desc"
    fi
  else
    fail=$((fail + 1)); echo "FAIL  - $desc (expected non-zero exit, got 0)"; echo "$out" | sed 's/^/        /'
  fi
}

# --- 1: approved maintainer identity -> PASS --------------------------------
new_repo; base=$(sha)
commit "wavicles" "praveenkumar103@gmail.com" "wavicles" "praveenkumar103@gmail.com" "maintainer commit"
assert_pass "approved maintainer identity" "$base" "$(sha)"
rm_repo

# --- 2: arbitrary external human contributor -> PASS ------------------------
new_repo; base=$(sha)
commit "Synthetic Human" "synthetic.human@example.invalid" "Synthetic Human" "synthetic.human@example.invalid" "external commit"
assert_pass "arbitrary external human contributor" "$base" "$(sha)"
rm_repo

# --- 3: external contributor, GitHub noreply email -> PASS ------------------
new_repo; base=$(sha)
commit "Synthetic Human" "12345+synthetichuman@users.noreply.github.com" \
       "Synthetic Human" "12345+synthetichuman@users.noreply.github.com" "noreply email commit"
assert_pass "external contributor with GitHub noreply email" "$base" "$(sha)"
rm_repo

# --- 4: GitHub platform committer -> PASS -----------------------------------
new_repo; base=$(sha)
commit "Synthetic Human" "synthetic.human@example.invalid" "GitHub" "noreply@github.com" "web-UI edit"
assert_pass "GitHub platform committer" "$base" "$(sha)"
rm_repo

# --- 5: two legitimate human co-authors -> PASS -----------------------------
new_repo; base=$(sha)
commit "Synthetic Human One" "human.one@example.invalid" "Synthetic Human One" "human.one@example.invalid" \
  "paired work" "Co-authored-by: Synthetic Human Two <human.two@example.invalid>" \
               "Co-authored-by: Synthetic Human Three <human.three@example.invalid>"
assert_pass "two legitimate human co-authors" "$base" "$(sha)"
rm_repo

# --- 6: multiple external contributors across commits -> PASS ---------------
new_repo; base=$(sha)
commit "Synthetic Human A" "human.a@example.invalid" "Synthetic Human A" "human.a@example.invalid" "commit A"
commit "Synthetic Human B" "human.b@example.invalid" "Synthetic Human B" "human.b@example.invalid" "commit B"
assert_pass "multiple external contributors" "$base" "$(sha)"
rm_repo

# --- 7: explicit "Claude Code" author identity -> FAIL ----------------------
new_repo; base=$(sha)
commit "Claude Code" "ci@example.invalid" "Claude Code" "ci@example.invalid" "ai-authored commit"
assert_fail "explicit 'Claude Code' author identity" "$base" "$(sha)" "prohibited Author"
rm_repo

# --- 8: explicit AI/tool committer identity -> FAIL -------------------------
new_repo; base=$(sha)
commit "Synthetic Human" "synthetic.human@example.invalid" "GitHub Copilot" "copilot@example.invalid" "ai committer"
assert_fail "explicit AI/tool committer identity" "$base" "$(sha)" "prohibited Committer"
rm_repo

# --- 9: Co-authored-by: Claude Code <...> -> FAIL ---------------------------
new_repo; base=$(sha)
commit "Synthetic Human" "synthetic.human@example.invalid" "Synthetic Human" "synthetic.human@example.invalid" \
  "human commit with AI co-author" "Co-authored-by: Claude Code <synthetic@example.invalid>"
assert_fail "Co-authored-by: Claude Code" "$base" "$(sha)" "Co-authored-by: Claude Code"
rm_repo

# --- 10: another explicit tool/model attribution -> FAIL --------------------
new_repo; base=$(sha)
commit "Gemini CLI" "ci@example.invalid" "Gemini CLI" "ci@example.invalid" "another ai-authored commit"
assert_fail "explicit 'Gemini CLI' identity" "$base" "$(sha)" "prohibited Author"
rm_repo

# --- 11: legitimate human named Claude -> PASS ------------------------------
new_repo; base=$(sha)
commit "Claude" "claude.human@example.invalid" "Claude" "claude.human@example.invalid" "human named Claude"
assert_pass "legitimate human named Claude (bare first name)" "$base" "$(sha)"
rm_repo
new_repo; base=$(sha)
commit "Claude Martin" "claude.martin@example.invalid" "Claude Martin" "claude.martin@example.invalid" "human named Claude Martin"
assert_pass "legitimate human named Claude Martin" "$base" "$(sha)"
rm_repo

# --- 12: legitimate human, AI-company-style domain -> PASS ------------------
new_repo; base=$(sha)
commit "Jane Smith" "jane.smith@anthropic.com" "Jane Smith" "jane.smith@anthropic.com" "human at an AI company"
assert_pass "human contributor at an AI-company-style domain" "$base" "$(sha)"
rm_repo

# --- 13: commit claims maintainer identity incorrectly -> FAIL --------------
new_repo; base=$(sha)
commit "wavicles" "attacker@example.invalid" "wavicles" "attacker@example.invalid" "impersonation attempt"
assert_fail "commit claims 'wavicles' with wrong email" "$base" "$(sha)" "claims maintainer identity"
rm_repo
new_repo; base=$(sha)
commit "Synthetic Human" "synthetic.human@example.invalid" "Praveen (Gnovi)" "wrong@example.invalid" "committer impersonation"
assert_fail "committer claims 'Praveen (Gnovi)' with wrong email" "$base" "$(sha)" "claims maintainer identity"
rm_repo

# --- 14: multiple commits, one prohibited -> FAIL overall -------------------
new_repo; base=$(sha)
commit "Synthetic Human" "synthetic.human@example.invalid" "Synthetic Human" "synthetic.human@example.invalid" "ok commit"
commit "ChatGPT" "ci@example.invalid" "ChatGPT" "ci@example.invalid" "bad commit"
assert_fail "one prohibited commit among several fails the whole range" "$base" "$(sha)" "prohibited Author"
rm_repo

# --- 15: push-to-main range after a normal GitHub-style merge ---------------
new_repo; base=$(sha)
git -C "$_repo" checkout -qb feature
commit "Synthetic Human" "synthetic.human@example.invalid" "Synthetic Human" "synthetic.human@example.invalid" "feature work"
git -C "$_repo" checkout -q main
GIT_AUTHOR_NAME="wavicles" GIT_AUTHOR_EMAIL="praveenkumar103@gmail.com" \
GIT_COMMITTER_NAME="wavicles" GIT_COMMITTER_EMAIL="praveenkumar103@gmail.com" \
  git -C "$_repo" merge -q --no-ff -m "Merge branch 'feature'" feature
assert_pass "push-to-main range after a normal merge" "$base" "$(sha)"
rm_repo

# --- 16: PR commit range (feature branch tip, no merge) ---------------------
new_repo; base=$(sha)
git -C "$_repo" checkout -qb feature
commit "Synthetic Human A" "human.a@example.invalid" "Synthetic Human A" "human.a@example.invalid" "commit A"
commit "Synthetic Human B" "human.b@example.invalid" "Synthetic Human B" "human.b@example.invalid" "commit B"
assert_pass "PR commit range (feature branch, base..head)" "$base" "$(sha)"
rm_repo

# --- extra: GitHub Actions Job Summary + notice are produced ----------------
new_repo; base=$(sha)
commit "Synthetic Human" "synthetic.human@example.invalid" "Synthetic Human" "synthetic.human@example.invalid" "external commit"
head=$(sha)
summary_file=$(mktemp)
out=$(cd "$_repo" && GITHUB_ACTIONS=true GITHUB_STEP_SUMMARY="$summary_file" "$CHECK" "$base" "$head" 2>&1)
if grep -q '^## Authorship Review' "$summary_file" \
   && grep -q 'External contributor -- verify before merge' "$summary_file" \
   && printf '%s' "$out" | grep -q '::notice::External contributor detected'; then
  pass=$((pass + 1)); echo "ok    - GitHub Actions Job Summary + notice produced"
else
  fail=$((fail + 1)); echo "FAIL  - GitHub Actions Job Summary + notice produced"
  echo "  --- summary file ---"; sed 's/^/        /' "$summary_file"
  echo "  --- stdout/stderr ---"; printf '%s\n' "$out" | sed 's/^/        /'
fi
rm -f "$summary_file"
rm_repo

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
