#!/usr/bin/env bash
# Source from the launchers, or execute a separately downloaded copy to repair
# a checkout whose current launcher cannot find git. No sudo or system installs.

traffic_require_git() {
    if command -v git >/dev/null 2>&1; then
        return 0
    fi
    local git_path init_file module_name
    for git_path in /usr/bin/git /usr/local/bin/git /bin/git; do
        if [[ -x "$git_path" ]]; then
            export PATH="${git_path%/*}:$PATH"
            return 0
        fi
    done
    # Module commands are often shell functions absent from noninteractive SSH.
    if ! type module >/dev/null 2>&1; then
        for init_file in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash; do
            if [[ -r "$init_file" ]]; then
                # Vendor module init files may reference unset shell variables.
                local restore_nounset=0
                [[ $- == *u* ]] && restore_nounset=1
                set +u
                source "$init_file" || true
                (( restore_nounset == 0 )) || set -u
                type module >/dev/null 2>&1 && break
            fi
        done
    fi
    if type module >/dev/null 2>&1; then
        for module_name in git Git; do
            printf 'Trying HPC module: %s\n' "$module_name"
            if module load "$module_name" && command -v git >/dev/null 2>&1; then
                printf 'Git available: %s\n' "$(command -v git)"
                return 0
            fi
        done
        printf '\nGit is still unavailable. Available matching modules follow:\n' >&2
        module -t avail git Git 2>&1 || true
        printf '\nLoad the exact Git module shown by your HPC, or provide this output for setup.\n' >&2
    else
        printf '\nGit is missing and no environment-module command was found.\n' >&2
        printf 'Ask HPC support for the supported Git/module setup, or provide this output. No downloads were started.\n' >&2
    fi
    return 127
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -euo pipefail
    TRAFFIC_PROJECT_DIR="${1:-$PWD}"
    if [[ ! -d "$TRAFFIC_PROJECT_DIR/.git" ]]; then
        printf 'Expected an existing Git checkout at: %s\n' "$TRAFFIC_PROJECT_DIR" >&2
        exit 2
    fi
    if [[ -f "$TRAFFIC_PROJECT_DIR/config/hpc.local.sh" ]]; then
        source "$TRAFFIC_PROJECT_DIR/config/hpc.local.sh"
    fi
    traffic_require_git || exit 2
    if [[ "$(git -C "$TRAFFIC_PROJECT_DIR" remote get-url origin)" != 'https://github.com/rtqayyum/nature-llm.git' &&
          "$(git -C "$TRAFFIC_PROJECT_DIR" remote get-url origin)" != 'git@github.com:rtqayyum/nature-llm.git' ]]; then
        printf 'Origin is not rtqayyum/nature-llm; stopped without changing this checkout.\n' >&2
        exit 2
    fi
    if [[ -n "$(git -C "$TRAFFIC_PROJECT_DIR" status --porcelain --untracked-files=no)" ]]; then
        printf 'Tracked files have local edits; stopped without discarding them.\n' >&2
        exit 2
    fi
    git -C "$TRAFFIC_PROJECT_DIR" pull --ff-only
    # The normal updater now checks the acquisition lock before another update.
    exec bash "$TRAFFIC_PROJECT_DIR/update_and_run.sh"
fi
