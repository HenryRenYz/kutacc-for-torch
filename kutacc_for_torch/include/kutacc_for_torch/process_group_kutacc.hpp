#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>

#include <chrono>
#include <cstdint>
#include <vector>

#if defined(__GNUC__)
#define KUTACC_TORCH_API __attribute__((visibility("default")))
#else
#define KUTACC_TORCH_API
#endif

namespace kutacc_torch {

// C++ entry point for embedding the Kutacc c10d backend without Python.
KUTACC_TORCH_API c10::intrusive_ptr<c10d::Backend> create_process_group_kutacc(
    c10::intrusive_ptr<c10d::Store> store,
    int rank,
    int size,
    std::chrono::milliseconds timeout);

// Ragged single-output AllGather extension. Counts and displacements are in
// logical elements, matching the native Kutacc API.
KUTACC_TORCH_API c10::intrusive_ptr<c10d::Work> allgatherv_into_tensor(
    c10d::Backend& backend,
    at::Tensor& output,
    at::Tensor& input,
    const std::vector<int64_t>& counts,
    const std::vector<int64_t>& displacements = {});

} // namespace kutacc_torch

