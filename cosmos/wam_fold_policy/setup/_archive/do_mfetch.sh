#!/usr/bin/env bash
export http_proxy=http://localhost:29290 https_proxy=http://localhost:29290 no_proxy=localhost GIT_TERMINAL_PROMPT=0
DB=/root/.cache/uv/git-v0/db
DIR="$DB/aa22635ca1311618"
SHA=de56227b26ba56de885897361272513697de7dc6
URL=https://github.com/NVIDIA/Megatron-LM.git
rm -rf "$DIR"; mkdir -p "$DIR"; git init --bare "$DIR" >/dev/null
ok=0
# Try full fetch of the single commit (with history). If it keeps dropping, fall to partial clone.
for a in $(seq 1 30); do
  echo "megatron FULL attempt $a $(date +%H:%M:%S)"
  git -C "$DIR" -c http.version=HTTP/1.1 -c http.postBuffer=2097152000 fetch --force --update-head-ok "$URL" "+$SHA:refs/commit/$SHA" 2>&1 | tail -2
  if git -C "$DIR" rev-parse --verify "refs/commit/$SHA^{commit}" >/dev/null 2>&1; then echo "MEGATRON FULL OK attempt $a"; ok=1; break; fi
  sleep 6
done
if [ $ok -eq 0 ]; then
  echo "=== FULL failed; trying partial clone (blob:none) ==="
  rm -rf "$DIR"; mkdir -p "$DIR"; git init --bare "$DIR" >/dev/null
  git -C "$DIR" remote add origin "$URL"
  git -C "$DIR" config remote.origin.promisor true
  git -C "$DIR" config remote.origin.partialclonefilter blob:none
  for a in $(seq 1 30); do
    echo "megatron PARTIAL attempt $a $(date +%H:%M:%S)"
    git -C "$DIR" -c http.version=HTTP/1.1 fetch --filter=blob:none --force --update-head-ok "$URL" "+$SHA:refs/commit/$SHA" 2>&1 | tail -2
    if git -C "$DIR" rev-parse --verify "refs/commit/$SHA" >/dev/null 2>&1; then echo "MEGATRON PARTIAL OK attempt $a"; ok=1; break; fi
    sleep 6
  done
fi
echo "=== MFETCH DONE ok=$ok ==="
