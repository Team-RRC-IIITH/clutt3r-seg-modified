#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${1:-external/DuoduoCLIP}"
REPO_URL="${DUODUOCLIP_REPO:-https://github.com/3dlg-hcvc/DuoduoCLIP.git}"
REF="${DUODUOCLIP_REF:-main}"

if [[ ! -d "${INSTALL_DIR}/.git" ]]; then
    mkdir -p "$(dirname "${INSTALL_DIR}")"
    git clone "${REPO_URL}" "${INSTALL_DIR}"
fi

git -C "${INSTALL_DIR}" fetch --tags origin
git -C "${INSTALL_DIR}" checkout "${REF}"

python -m pip install -r "${INSTALL_DIR}/requirements.txt"
python -m pip install -e "${INSTALL_DIR}/open_clip_mod"

ABS_INSTALL_DIR="$(cd "${INSTALL_DIR}" && pwd)"
echo "DuoduoCLIP is installed at ${ABS_INSTALL_DIR}"
echo "Set this before running Clutt3R-Seg:"
echo "export DUODUOCLIP_ROOT=${ABS_INSTALL_DIR}"
