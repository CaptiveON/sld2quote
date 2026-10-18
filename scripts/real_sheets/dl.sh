#!/bin/zsh
# usage: dl.sh <source-id> <url> [filename]   -> data/real/raw/<source-id>/<filename> + note.txt
set -e
R="$(cd "$(dirname "$0")/../.." && pwd)/data/real/raw"
id=$1; url=$2; name=${3:-${url:t}}
d=$R/$id; mkdir -p $d
out=$(curl -sS -L --fail --max-time 300 -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15" \
  -o "$d/$name" -w '%{http_code} %{content_type} %{url_effective}' "$url")
{
  echo "url: $url"
  echo "final_url: ${out#* * }"
  echo "http: ${out%% *}  content_type: $(echo $out | cut -d' ' -f2)"
  echo "retrieved_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "file: $name  bytes: $(stat -f %z "$d/$name")  sha256: $(shasum -a 256 "$d/$name" | cut -d' ' -f1)"
} > $d/note.txt
cat $d/note.txt; file -b "$d/$name"
