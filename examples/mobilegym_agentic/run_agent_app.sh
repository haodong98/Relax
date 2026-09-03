#!/bin/bash

export OPENAI_BASE_URL="${RELAX_BASE_URL}"
export OPENAI_API_KEY="${RELAX_SESSION_ID}"
# MobileGym's screenshots are append-only trajectory evidence for the VLM
# judge.  Keep the entire interaction history unless a caller explicitly opts
# into a different experimental contract.
export MOBILEGYM_HISTORY_IMAGES="${MOBILEGYM_HISTORY_IMAGES:-1}"

python -m app.agent \
    --input-json "${RELAX_INPUT_JSON}" \
    --output-json "${RELAX_OUTPUT_JSON}"
