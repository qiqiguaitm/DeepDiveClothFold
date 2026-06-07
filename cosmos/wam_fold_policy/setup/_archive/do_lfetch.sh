#!/usr/bin/env bash
export http_proxy=http://localhost:29290 https_proxy=http://localhost:29290 no_proxy=localhost GIT_TERMINAL_PROMPT=0
DB=/root/.cache/uv/git-v0/db
DIR="$DB/3854d3ea6f0ea07f"
SHA=1a4316c6845330bc552fb982dbc44bdb4f66f2f1
URL=https://github.com/mli0603/lerobot.git
rm -rf "$DIR"; mkdir -p "$DIR"; git init --bare "$DIR" >/dev/null
for a in $(seq 1 40); do
  echo "lerobot full fetch attempt $a $(date +%H:%M:%S)"
  git -C "$DIR" -c http.version=HTTP/1.1 -c http.postBuffer=1048576000 fetch --force --update-head-ok "$URL" "+$SHA:refs/commit/$SHA" 2>&1 | tail -2
  if git -C "$DIR" rev-parse --verify "refs/commit/$SHA^{commit}" >/dev/null 2>&1; then echo "LEROBOT FULL OK attempt $a"; break; fi
  sleep 6
done
echo "=== LFETCH DONE ==="
