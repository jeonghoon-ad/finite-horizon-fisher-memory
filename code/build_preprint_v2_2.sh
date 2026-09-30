#!/bin/sh
# Build the v2.2 public manuscript (2026-09-23). See provenance/CHANGELOG_v2.0_to_v2.1_EN.md and CHANGELOG_v2.1_to_v2.2_EN.md. Recorded so the TeX and PDF are reproducible from the
# markdown source, per review item R5 and the clean-build verification.
#
#   pandoc 3.11, Tectonic 0.16.9 (XeTeX)
#
# Title, subtitle, author and date come from the markdown YAML metadata, not from this
# script, so the typeset front matter cannot drift from the source.
#
# Runs in two layouts: the lane, where this script sits beside the manuscript in
# docs/v2_0/, and the release candidate, where it sits in code/ and the manuscript is in
# ../manuscript/. Figures are resolved relative to the manuscript, and are rendered by
# make_public_figures.py (all seven manuscript figures).
set -eu

# Pin the timestamp TeX and xdvipdfmx write into the PDF, so the same source produces the
# same PDF bytes on any machine and the recorded sha256 is checkable rather than merely
# descriptive. 1790726400 is 2026-09-30T00:00:00Z, this build's date.
SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-1790726400}
FORCE_SOURCE_DATE=1
export SOURCE_DATE_EPOCH FORCE_SOURCE_DATE

BASE=PREPRINT_PUBLIC_BUILD_v2.2_20260923_EN
HERE=$(cd "$(dirname "$0")" && pwd)

if [ -f "$HERE/$BASE.md" ]; then
  SRC="$HERE"
elif [ -f "$HERE/../manuscript/$BASE.md" ]; then
  SRC=$(cd "$HERE/../manuscript" && pwd)
else
  echo "build_preprint_v2_2.sh: cannot find $BASE.md next to the script or in ../manuscript" >&2
  exit 1
fi

if [ -f "$HERE/preprint_header_v2_0.tex" ]; then
  HEADER="$HERE/preprint_header_v2_0.tex"
elif [ -f "$SRC/preprint_header_v2_0.tex" ]; then
  HEADER="$SRC/preprint_header_v2_0.tex"
else
  echo "build_preprint_v2_2.sh: cannot find preprint_header_v2_0.tex" >&2
  exit 1
fi

cd "$SRC"

pandoc "$BASE.md" -s -o "$BASE.tex" \
  -f markdown+tex_math_single_backslash \
  --pdf-engine=xelatex \
  -V fontsize=10pt \
  -V geometry:margin=1in \
  --lua-filter "$HERE/abstract_layout.lua" \
  -H "$HEADER"

tectonic -X compile "$BASE.tex" --outdir . --keep-logs --print 2>&1 | tee tectonic_build.log

# A reader-facing page must never lose text off the right margin.
OVERFULL=$(grep -c "Overfull" "$BASE.log" || true)
echo "overfull boxes: $OVERFULL"
[ "$OVERFULL" -eq 0 ] || { echo "build_preprint_v2_2.sh: $OVERFULL overfull boxes" >&2; exit 1; }
