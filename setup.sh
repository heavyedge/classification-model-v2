#!/bin/sh

pip install uv

uv tool install --force 'huggingface_hub[cli]'
export PATH="$(uv tool dir --bin):$PATH"
export HF_TOKEN="${HF_TOKEN:-$HUGGINGFACE_TOKEN}"

mkdir -p ./_data/v2/

(
    uv pip install --system -r requirements.txt -r examples/requirements.txt
) &
requirements_pid=$!

(
    curl -LsSf https://hf.co/cli/install.sh | bash
    hf auth login --token "$HUGGINGFACE_TOKEN"
    hf download heavyedge/profiles --repo-type dataset --revision v2.0.0 --include "v2/profiles/mean_profiles/*.tar.gz" --local-dir _data/
    for dataset in _data/v2/profiles/mean_profiles/*.tar.gz; do
        stem=$(basename "$dataset" .tar.gz)
        dirname=_data/v2/profiles/mean_profiles/"$stem"
        mkdir -p "$dirname"
        tar -xzf "$dataset" -C "$dirname"
    done
    rm -f _data/v2/profiles/mean_profiles/*.tar.gz
) &
profiles_pid=$!

(
  uv pip install --system 'gdown<6.0.0'
  gdown --fuzzy "$LABELS_V1_GDRIVE" -O ./_data/v2/labels.tar
  mkdir -p ./_data/v2/labels
  tar -xf _data/v2/labels.tar -C _data/v2/labels
) &
labels_pid=$!

wait "$requirements_pid"
wait "$profiles_pid"
wait "$labels_pid"
