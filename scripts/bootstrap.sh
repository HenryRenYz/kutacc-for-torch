#!/usr/bin/env bash

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under a modified version of the MIT license. See LICENSE in the project root for license information.

# Fetch the vendored kutacc/kupl sources pinned as git submodules.
#
# The pinned commits live in the parent repository's index; this script makes
# third_party/{kupl,kutacc} contain exactly those commits.  When a submodule
# URL is not reachable (for example the kutacc fork does not carry the pinned
# feature branch yet), point the corresponding *_MIRROR variable at any git
# repository or local directory that contains the commit:
#
#   KUTACC_FOR_TORCH_KUPL_MIRROR=/path/to/kupl ./scripts/bootstrap.sh
#   KUTACC_FOR_TORCH_KUTACC_MIRROR=https://.../kutacc.git ./scripts/bootstrap.sh
set -euo pipefail

cd "$(cd "$(dirname "$0")/.." && pwd)"

ensure_dep() {
    local name=$1
    local mirror=$2
    local dir="third_party/$name"

    if [ -f "$dir/CMakeLists.txt" ]; then
        echo "$name: sources already present"
        return
    fi

    local sha
    sha=$(git ls-tree HEAD "third_party/$name" | awk '{print $3}')
    if [ -z "$sha" ]; then
        echo "error: $name is not pinned in the repository index" >&2
        return 1
    fi

    if [ -n "$mirror" ]; then
        echo "$name: cloning from mirror $mirror at $sha"
        git -c protocol.file.allow=always clone "$mirror" "$dir"
        git -C "$dir" checkout --detach "$sha"
    else
        echo "$name: updating submodule from $(git config -f .gitmodules submodule.$name.url)"
        git -c protocol.file.allow=always submodule update --init --recursive "third_party/$name"
    fi
}

ensure_dep kupl "${KUTACC_FOR_TORCH_KUPL_MIRROR:-}"
ensure_dep kutacc "${KUTACC_FOR_TORCH_KUTACC_MIRROR:-}"

echo "vendored sources ready:"
git submodule status
