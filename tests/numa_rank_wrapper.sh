#!/usr/bin/env bash

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under a modified version of the MIT license. See LICENSE in the project root for license information.

set -euo pipefail

rank="${OMPI_COMM_WORLD_RANK:?OMPI_COMM_WORLD_RANK is required}"
world_size="${OMPI_COMM_WORLD_SIZE:?OMPI_COMM_WORLD_SIZE is required}"
local_rank="${OMPI_COMM_WORLD_LOCAL_RANK:-$rank}"
numa_count="${KUTACC_TEST_NUMA_COUNT:-16}"
cpu_node=$((local_rank % numa_count))

case "${KUTACC_TEST_MEMORY_KIND:-dram}" in
    dram) memory_node="$cpu_node" ;;
    hbm) memory_node=$((cpu_node + ${KUTACC_TEST_HBM_OFFSET:-16})) ;;
    *) echo "unsupported KUTACC_TEST_MEMORY_KIND=${KUTACC_TEST_MEMORY_KIND}" >&2; exit 2 ;;
esac

export RANK="$rank"
export WORLD_SIZE="$world_size"
export LOCAL_RANK="$local_rank"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

exec numactl --cpunodebind="$cpu_node" --membind="$memory_node" "$@"

