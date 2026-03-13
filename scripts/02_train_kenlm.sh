#!/bin/bash
# Step 2: Train a KenLM n-gram language model on prepared corpus.
# Prerequisites: KenLM installed (https://github.com/kpu/kenlm)
#
# Usage: bash scripts/02_train_kenlm.sh [corpus_file] [output_dir] [n]
#
# Arguments:
#   corpus_file: one-sentence-per-line tokenized text file (default: data/lm_corpus.txt)
#   output_dir:  where to save the .arpa and .binary files (default: lm/)
#   n:           n-gram order (default: 5)

set -e

CORPUS="${1:-data/lm_corpus.txt}"
OUTPUT_DIR="${2:-lm}"
N="${3:-5}"

mkdir -p "$OUTPUT_DIR"

if ! command -v lmplz &> /dev/null; then
    echo "ERROR: lmplz not found. Please install KenLM: https://github.com/kpu/kenlm"
    echo "Quick install:"
    echo "  git clone https://github.com/kpu/kenlm.git"
    echo "  mkdir kenlm/build && cd kenlm/build"
    echo "  cmake .. && make -j4"
    echo "  sudo make install"
    exit 1
fi

ARPA="$OUTPUT_DIR/model_${N}gram.arpa"
BIN="$OUTPUT_DIR/model_${N}gram.binary"

echo "Training ${N}-gram KenLM on: $CORPUS"
lmplz -o "$N" --text "$CORPUS" --arpa "$ARPA" --discount_fallback

echo "Binarizing..."
build_binary "$ARPA" "$BIN"

echo "Done! Model saved to $BIN"
