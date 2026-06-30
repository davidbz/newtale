#!/usr/bin/env bash

set -Eeuo pipefail

readonly CLAUDE_INSTALL_URL="https://claude.ai/install.sh"
readonly HEADROOM_PACKAGE="headroom-ai[all]"

log() {
    printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

require() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "error: missing dependency: $1" >&2
        exit 1
    }
}

main() {
    require curl
    require python3
    require pip

    log "Installing Claude CLI..."
    curl -fsSL "$CLAUDE_INSTALL_URL" | bash

    log "Installing Headroom AI..."
    python3 -m pip install --upgrade "$HEADROOM_PACKAGE"

    log "Done."

    cat <<'EOF'

Next steps:
  headroom wrap claude
  /plugin marketplace add DietrichGebert/ponytail
  /plugin install ponytail@ponytail

EOF
}

main "$@"
