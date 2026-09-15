/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * Licensed under a modified version of the MIT license. See LICENSE in the project root for license information.
 */

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <kutacc_for_torch/process_group_kutacc.hpp>
#include <kutacc.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/utils/pybind.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <unistd.h>

namespace kutacc_torch {
namespace {

size_t datatype_size(kupl_shm_datatype_t datatype) {
  switch (datatype) {
    case KUPL_SHM_DATATYPE_CHAR:
      return sizeof(char);
    case KUPL_SHM_DATATYPE_INT:
      return sizeof(int);
    case KUPL_SHM_DATATYPE_LONG:
      return sizeof(long);
    case KUPL_SHM_DATATYPE_FLOAT:
      return sizeof(float);
    case KUPL_SHM_DATATYPE_DOUBLE:
      return sizeof(double);
    default:
      return 0;
  }
}

size_t checked_size(int count, size_t element_size) {
  TORCH_CHECK(count >= 0, "Kutacc OOB received a negative count");
  TORCH_CHECK(element_size != 0, "Kutacc OOB received an unsupported datatype");
  TORCH_CHECK(
      element_size == 0 || static_cast<size_t>(count) <= std::numeric_limits<size_t>::max() / element_size,
      "Kutacc OOB byte count overflow");
  return static_cast<size_t>(count) * element_size;
}

void throw_on_status(kutacc::shm_comm_status_t status, const char* operation) {
  TORCH_CHECK(
      status == kutacc::SHM_COMM_SUCCESS,
      "Kutacc ",
      operation,
      " failed: ",
      kutacc::shm_comm_status_string(status));
}

struct OobContext {
  c10::intrusive_ptr<c10d::Store> store;
  int rank;
  int size;
  std::chrono::milliseconds timeout;
  std::atomic<uint64_t> allgather_sequence{0};
  std::atomic<uint64_t> barrier_sequence{0};

  std::string key(const char* operation, uint64_t sequence, int peer) const {
    return std::string("kutacc/oob/") + operation + "/" + std::to_string(sequence) + "/" +
        std::to_string(peer);
  }
};

int oob_allgather(
    const void* send_buffer,
    void* receive_buffer,
    int count,
    void* opaque,
    kupl_shm_datatype_t datatype) {
  auto* context = static_cast<OobContext*>(opaque);
  try {
    TORCH_CHECK(context != nullptr, "Kutacc OOB context is null");
    const size_t bytes = checked_size(count, datatype_size(datatype));
    TORCH_CHECK(bytes == 0 || send_buffer != nullptr, "Kutacc OOB send buffer is null");
    TORCH_CHECK(bytes == 0 || receive_buffer != nullptr, "Kutacc OOB receive buffer is null");
    const uint64_t sequence = context->allgather_sequence.fetch_add(1, std::memory_order_relaxed);
    std::vector<uint8_t> local(bytes);
    if (bytes != 0) {
      std::memcpy(local.data(), send_buffer, bytes);
    }
    context->store->set(context->key("allgather", sequence, context->rank), local);
    std::vector<std::string> keys;
    keys.reserve(context->size);
    for (int peer = 0; peer < context->size; ++peer) {
      keys.push_back(context->key("allgather", sequence, peer));
    }
    context->store->wait(keys, context->timeout);
    for (int peer = 0; peer < context->size; ++peer) {
      std::vector<uint8_t> value = context->store->get(keys[peer]);
      TORCH_CHECK(value.size() == bytes, "Kutacc OOB AllGather size mismatch");
      if (bytes != 0) {
        std::memcpy(static_cast<uint8_t*>(receive_buffer) + static_cast<size_t>(peer) * bytes, value.data(), bytes);
      }
    }
    return KUPL_OK;
  } catch (...) {
    return KUPL_ERROR;
  }
}

int oob_barrier(void* opaque) {
  auto* context = static_cast<OobContext*>(opaque);
  try {
    TORCH_CHECK(context != nullptr, "Kutacc OOB context is null");
    const uint64_t sequence = context->barrier_sequence.fetch_add(1, std::memory_order_relaxed);
    const std::vector<uint8_t> value{1};
    context->store->set(context->key("barrier", sequence, context->rank), value);
    std::vector<std::string> keys;
    keys.reserve(context->size);
    for (int peer = 0; peer < context->size; ++peer) {
      keys.push_back(context->key("barrier", sequence, peer));
    }
    context->store->wait(keys, context->timeout);
    return KUPL_OK;
  } catch (...) {
    return KUPL_ERROR;
  }
}

struct ConstView {
  std::vector<int64_t> sizes;
  std::vector<int64_t> byte_strides;
  kutacc::shm_const_tensor_view_t view{};
};

struct MutableView {
  std::vector<int64_t> sizes;
  std::vector<int64_t> byte_strides;
  kutacc::shm_tensor_view_t view{};
};

void check_tensor(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.device().is_cpu(), "ProcessGroupKutacc supports CPU tensors only");
  TORCH_CHECK(tensor.layout() == c10::kStrided, "ProcessGroupKutacc supports strided tensors only");
  TORCH_CHECK(tensor.element_size() > 0, "ProcessGroupKutacc received a tensor with zero-sized elements");
}

std::vector<int64_t> byte_strides(const at::Tensor& tensor) {
  std::vector<int64_t> result;
  result.reserve(tensor.dim());
  const int64_t element_size = static_cast<int64_t>(tensor.element_size());
  for (int64_t dimension = 0; dimension < tensor.dim(); ++dimension) {
    const int64_t stride = tensor.stride(dimension);
    TORCH_CHECK(stride >= 0, "ProcessGroupKutacc does not support negative tensor strides");
    TORCH_CHECK(
        stride == 0 || stride <= std::numeric_limits<int64_t>::max() / element_size,
        "ProcessGroupKutacc byte stride overflow");
    result.push_back(stride * element_size);
  }
  return result;
}

ConstView make_const_view(const at::Tensor& tensor) {
  check_tensor(tensor);
  ConstView result;
  result.sizes.assign(tensor.sizes().begin(), tensor.sizes().end());
  result.byte_strides = byte_strides(tensor);
  result.view = {tensor.numel() == 0 ? nullptr : tensor.const_data_ptr(),
                 static_cast<size_t>(tensor.element_size()),
                 tensor.dim(),
                 result.sizes.data(),
                 result.byte_strides.data(),
                 0};
  return result;
}

MutableView make_mutable_view(at::Tensor& tensor) {
  check_tensor(tensor);
  MutableView result;
  result.sizes.assign(tensor.sizes().begin(), tensor.sizes().end());
  result.byte_strides = byte_strides(tensor);
  result.view = {tensor.numel() == 0 ? nullptr : tensor.mutable_data_ptr(),
                 static_cast<size_t>(tensor.element_size()),
                 tensor.dim(),
                 result.sizes.data(),
                 result.byte_strides.data(),
                 0};
  return result;
}

std::vector<size_t> checked_sizes(const std::vector<int64_t>& values, const char* name) {
  std::vector<size_t> result;
  result.reserve(values.size());
  for (int64_t value : values) {
    TORCH_CHECK(value >= 0, name, " contains a negative value");
    result.push_back(static_cast<size_t>(value));
  }
  return result;
}

std::vector<size_t> prefix_displacements(const std::vector<size_t>& counts) {
  std::vector<size_t> result(counts.size(), 0);
  for (size_t index = 1; index < counts.size(); ++index) {
    TORCH_CHECK(
        result[index - 1] <= std::numeric_limits<size_t>::max() - counts[index - 1],
        "ProcessGroupKutacc displacement overflow");
    result[index] = result[index - 1] + counts[index - 1];
  }
  return result;
}

std::vector<size_t> split_element_counts(
    const at::Tensor& tensor,
    const std::vector<int64_t>& splits,
    int world_size,
    const char* name) {
  TORCH_CHECK(tensor.dim() > 0, name, " requires a tensor with at least one dimension");
  std::vector<int64_t> rows = splits;
  if (rows.empty()) {
    TORCH_CHECK(tensor.size(0) % world_size == 0, name, " cannot split dimension zero evenly");
    rows.assign(world_size, tensor.size(0) / world_size);
  }
  TORCH_CHECK(rows.size() == static_cast<size_t>(world_size), name, " must have world_size entries");
  int64_t row_sum = 0;
  for (int64_t rows_for_peer : rows) {
    TORCH_CHECK(rows_for_peer >= 0, name, " contains a negative split");
    TORCH_CHECK(row_sum <= std::numeric_limits<int64_t>::max() - rows_for_peer, name, " sum overflow");
    row_sum += rows_for_peer;
  }
  TORCH_CHECK(row_sum == tensor.size(0), name, " must sum to tensor.size(0)");

  size_t elements_per_row = 1;
  for (int64_t dimension = 1; dimension < tensor.dim(); ++dimension) {
    TORCH_CHECK(
        tensor.size(dimension) == 0 ||
            elements_per_row <= std::numeric_limits<size_t>::max() / static_cast<size_t>(tensor.size(dimension)),
        name,
        " element count overflow");
    elements_per_row *= static_cast<size_t>(tensor.size(dimension));
  }
  std::vector<size_t> result;
  result.reserve(rows.size());
  for (int64_t rows_for_peer : rows) {
    TORCH_CHECK(
        rows_for_peer == 0 || elements_per_row <= std::numeric_limits<size_t>::max() / rows_for_peer,
        name,
        " element count overflow");
    result.push_back(elements_per_row * static_cast<size_t>(rows_for_peer));
  }
  return result;
}

void configure_copy_threads_from_torch() {
  if (std::getenv("KUTACC_SHM_COPY_THREADS") != nullptr) {
    return;
  }
  int threads = std::max(1, at::get_num_threads());
  int maximum = 8;
  if (const char* configured = std::getenv("KUTACC_TORCH_MAX_COPY_THREADS")) {
    char* end = nullptr;
    long parsed = std::strtol(configured, &end, 10);
    if (end != configured && *end == '\0' && parsed > 0 && parsed <= std::numeric_limits<int>::max()) {
      maximum = static_cast<int>(parsed);
    }
  }
  threads = std::min(threads, maximum);
  const std::string value = std::to_string(threads);
  setenv("KUTACC_SHM_COPY_THREADS", value.c_str(), 0);
  setenv("KUTACC_SHM_COPY_POOL_THREADS", value.c_str(), 0);
}

} // namespace

class WorkKutacc final : public c10d::Work {
 public:
  WorkKutacc(
      int rank,
      c10d::OpType operation_type,
      kutacc::shm_comm_work_h native_work,
      std::vector<at::Tensor> result_tensors,
      std::vector<at::Tensor> retained_tensors)
      : Work(rank, operation_type),
        native_work_(native_work),
        result_tensors_(std::move(result_tensors)),
        retained_tensors_(std::move(retained_tensors)) {}

  ~WorkKutacc() override {
    if (native_work_ != nullptr) {
      const auto status = kutacc::shm_comm_work_destroy(native_work_);
      complete(status);
    }
  }

  bool isCompleted() override {
    if (native_work_ != nullptr) {
      bool completed = false;
      kutacc::shm_comm_status_t operation_status = kutacc::SHM_COMM_SUCCESS;
      const auto query_status = kutacc::shm_comm_work_test(native_work_, completed, operation_status);
      if (query_status != kutacc::SHM_COMM_SUCCESS) {
        complete(query_status);
      } else if (completed) {
        complete(operation_status);
      }
    }
    return c10d::Work::isCompleted();
  }

  bool wait(std::chrono::milliseconds timeout = kNoTimeout) override {
    if (native_work_ != nullptr) {
      if (timeout == kNoTimeout) {
        complete(kutacc::shm_comm_work_wait(native_work_));
      } else {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (!isCompleted()) {
          TORCH_CHECK(std::chrono::steady_clock::now() < deadline, "ProcessGroupKutacc operation timed out");
          std::this_thread::yield();
        }
      }
    }
    return c10d::Work::wait(kNoTimeout);
  }

  std::vector<at::Tensor> result() override {
    return result_tensors_;
  }

 private:
  void complete(kutacc::shm_comm_status_t status) noexcept {
    std::call_once(completed_once_, [this, status]() {
      if (status == kutacc::SHM_COMM_SUCCESS) {
        finish();
      } else {
        finish(std::make_exception_ptr(std::runtime_error(
            std::string("Kutacc collective failed: ") + kutacc::shm_comm_status_string(status))));
      }
    });
  }

  kutacc::shm_comm_work_h native_work_ = nullptr;
  std::vector<at::Tensor> result_tensors_;
  std::vector<at::Tensor> retained_tensors_;
  std::once_flag completed_once_;
};

class ProcessGroupKutacc final : public c10d::Backend {
 public:
  ProcessGroupKutacc(
      c10::intrusive_ptr<c10d::Store> store,
      int rank,
      int size,
      std::chrono::milliseconds timeout)
      : Backend(rank, size), oob_{std::move(store), rank, size, timeout} {
    TORCH_CHECK(size > 0 && rank >= 0 && rank < size, "invalid Kutacc rank or world size");
    configure_copy_threads_from_torch();
    kupl_shm_oob_cb_t callbacks{oob_allgather, oob_barrier};
    TORCH_CHECK(
        kupl_shm_comm_create(size, rank, static_cast<int>(getpid()), &callbacks, &oob_, &communicator_) == KUPL_OK,
        "failed to create the Kutacc KUPL communicator");
    const auto status = kutacc::shm_comm_context_create(communicator_, 0, context_);
    if (status != kutacc::SHM_COMM_SUCCESS) {
      kupl_shm_comm_destroy(communicator_);
      communicator_ = nullptr;
      throw_on_status(status, "context creation");
    }
  }

  ~ProcessGroupKutacc() override {
    if (context_ != nullptr) {
      kutacc::shm_comm_context_destroy(context_);
    }
    if (communicator_ != nullptr) {
      kupl_shm_comm_destroy(communicator_);
      communicator_ = nullptr;
    }
  }

  const std::string getBackendName() const override {
    return "kutacc";
  }

  c10::intrusive_ptr<c10d::Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const c10d::BroadcastOptions& options = c10d::BroadcastOptions()) override {
    TORCH_CHECK(tensors.size() == 1, "ProcessGroupKutacc Broadcast expects one CPU tensor");
    TORCH_CHECK(options.rootTensor == 0, "ProcessGroupKutacc Broadcast supports rootTensor == 0 only");
    MutableView buffer = make_mutable_view(tensors[0]);
    kutacc::shm_comm_work_h native_work = nullptr;
    throw_on_status(
        kutacc::shm_broadcast_async(context_, buffer.view, options.rootRank, native_work), "Broadcast submission");
    return make_work(c10d::OpType::BROADCAST, native_work, tensors, tensors);
  }

  c10::intrusive_ptr<c10d::Work> allgather(
      std::vector<std::vector<at::Tensor>>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::AllgatherOptions& = c10d::AllgatherOptions()) override {
    TORCH_CHECK(input_tensors.size() == 1, "ProcessGroupKutacc AllGather expects one CPU input tensor");
    TORCH_CHECK(output_tensors.size() == 1, "ProcessGroupKutacc AllGather expects one output tensor list");
    TORCH_CHECK(
        output_tensors[0].size() == static_cast<size_t>(getSize()),
        "ProcessGroupKutacc AllGather output list must have world_size entries");
    ConstView input = make_const_view(input_tensors[0]);
    std::vector<MutableView> owned_outputs;
    std::vector<kutacc::shm_tensor_view_t> outputs;
    owned_outputs.reserve(getSize());
    outputs.reserve(getSize());
    for (at::Tensor& tensor : output_tensors[0]) {
      owned_outputs.push_back(make_mutable_view(tensor));
      outputs.push_back(owned_outputs.back().view);
    }
    kutacc::shm_comm_work_h native_work = nullptr;
    throw_on_status(
        kutacc::shm_allgather_async(context_, input.view, outputs.data(), outputs.size(), native_work),
        "AllGather submission");
    std::vector<at::Tensor> retained = input_tensors;
    retained.insert(retained.end(), output_tensors[0].begin(), output_tensors[0].end());
    return make_work(c10d::OpType::ALLGATHER, native_work, output_tensors[0], std::move(retained));
  }

  c10::intrusive_ptr<c10d::Work> _allgather_base(
      at::Tensor& output,
      at::Tensor& input,
      const c10d::AllgatherOptions& = c10d::AllgatherOptions()) override {
    ConstView input_view = make_const_view(input);
    MutableView output_view = make_mutable_view(output);
    kutacc::shm_comm_work_h native_work = nullptr;
    throw_on_status(
        kutacc::shm_allgather_single_async(context_, input_view.view, output_view.view, native_work),
        "AllGather single submission");
    return make_work(c10d::OpType::_ALLGATHER_BASE, native_work, {output}, {input, output});
  }

  c10::intrusive_ptr<c10d::Work> alltoall(
      std::vector<at::Tensor>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::AllToAllOptions& = c10d::AllToAllOptions()) override {
    TORCH_CHECK(
        input_tensors.size() == static_cast<size_t>(getSize()) &&
            output_tensors.size() == static_cast<size_t>(getSize()),
        "ProcessGroupKutacc AllToAll tensor lists must have world_size entries");
    std::vector<ConstView> owned_inputs;
    std::vector<MutableView> owned_outputs;
    std::vector<kutacc::shm_const_tensor_view_t> inputs;
    std::vector<kutacc::shm_tensor_view_t> outputs;
    owned_inputs.reserve(getSize());
    owned_outputs.reserve(getSize());
    inputs.reserve(getSize());
    outputs.reserve(getSize());
    for (at::Tensor& tensor : input_tensors) {
      owned_inputs.push_back(make_const_view(tensor));
      inputs.push_back(owned_inputs.back().view);
    }
    for (at::Tensor& tensor : output_tensors) {
      owned_outputs.push_back(make_mutable_view(tensor));
      outputs.push_back(owned_outputs.back().view);
    }
    kutacc::shm_comm_work_h native_work = nullptr;
    throw_on_status(
        kutacc::shm_alltoall_async(
            context_, inputs.data(), inputs.size(), outputs.data(), outputs.size(), native_work),
        "AllToAll submission");
    std::vector<at::Tensor> retained = input_tensors;
    retained.insert(retained.end(), output_tensors.begin(), output_tensors.end());
    return make_work(c10d::OpType::ALLTOALL, native_work, output_tensors, std::move(retained));
  }

  c10::intrusive_ptr<c10d::Work> alltoall_base(
      at::Tensor& output,
      at::Tensor& input,
      std::vector<int64_t>& output_split_sizes,
      std::vector<int64_t>& input_split_sizes,
      const c10d::AllToAllOptions& = c10d::AllToAllOptions()) override {
    ConstView input_view = make_const_view(input);
    MutableView output_view = make_mutable_view(output);
    std::vector<size_t> send_counts =
        split_element_counts(input, input_split_sizes, getSize(), "inputSplitSizes");
    std::vector<size_t> receive_counts =
        split_element_counts(output, output_split_sizes, getSize(), "outputSplitSizes");
    std::vector<size_t> send_displacements = prefix_displacements(send_counts);
    std::vector<size_t> receive_displacements = prefix_displacements(receive_counts);
    kutacc::shm_comm_work_h native_work = nullptr;
    throw_on_status(
        kutacc::shm_alltoall_single_async(
            context_,
            input_view.view,
            output_view.view,
            send_counts.data(),
            send_displacements.data(),
            receive_counts.data(),
            receive_displacements.data(),
            native_work),
        "AllToAll single submission");
    return make_work(c10d::OpType::ALLTOALL_BASE, native_work, {output}, {input, output});
  }

  c10::intrusive_ptr<c10d::Work> barrier(
      const c10d::BarrierOptions& = c10d::BarrierOptions()) override {
    at::Tensor token = at::empty({0}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    MutableView token_view = make_mutable_view(token);
    kutacc::shm_comm_work_h native_work = nullptr;
    throw_on_status(kutacc::shm_broadcast_async(context_, token_view.view, 0, native_work), "Barrier submission");
    return make_work(c10d::OpType::BARRIER, native_work, {}, {token});
  }

  c10::intrusive_ptr<c10d::Work> allgatherv_into_tensor(
      at::Tensor& output,
      at::Tensor& input,
      const std::vector<int64_t>& counts,
      const std::vector<int64_t>& displacements) {
    TORCH_CHECK(counts.size() == static_cast<size_t>(getSize()), "counts must have world_size entries");
    std::vector<size_t> native_counts = checked_sizes(counts, "counts");
    std::vector<size_t> native_displacements =
        displacements.empty() ? prefix_displacements(native_counts) : checked_sizes(displacements, "displacements");
    TORCH_CHECK(
        native_displacements.size() == static_cast<size_t>(getSize()),
        "displacements must have world_size entries");
    ConstView input_view = make_const_view(input);
    MutableView output_view = make_mutable_view(output);
    kutacc::shm_comm_work_h native_work = nullptr;
    throw_on_status(
        kutacc::shm_allgatherv_single_async(
            context_,
            input_view.view,
            output_view.view,
            native_counts.data(),
            native_displacements.data(),
            native_work),
        "AllGatherV submission");
    return make_work(c10d::OpType::_ALLGATHER_BASE, native_work, {output}, {input, output});
  }

 private:
  c10::intrusive_ptr<c10d::Work> make_work(
      c10d::OpType operation_type,
      kutacc::shm_comm_work_h native_work,
      std::vector<at::Tensor> result_tensors,
      std::vector<at::Tensor> retained_tensors) {
    return c10::make_intrusive<WorkKutacc>(
        getRank(), operation_type, native_work, std::move(result_tensors), std::move(retained_tensors));
  }

  OobContext oob_;
  kupl_shm_comm_h communicator_ = nullptr;
  kutacc::shm_comm_context_h context_ = nullptr;
};

c10::intrusive_ptr<c10d::Backend> create_process_group_kutacc(
    c10::intrusive_ptr<c10d::Store> store,
    int rank,
    int size,
    std::chrono::milliseconds timeout) {
  return c10::make_intrusive<ProcessGroupKutacc>(std::move(store), rank, size, timeout);
}

c10::intrusive_ptr<c10d::Work> allgatherv_into_tensor(
    c10d::Backend& backend,
    at::Tensor& output,
    at::Tensor& input,
    const std::vector<int64_t>& counts,
    const std::vector<int64_t>& displacements) {
  auto* kutacc_backend = dynamic_cast<ProcessGroupKutacc*>(&backend);
  TORCH_CHECK(kutacc_backend != nullptr, "backend is not a ProcessGroupKutacc instance");
  return kutacc_backend->allgatherv_into_tensor(output, input, counts, displacements);
}

} // namespace kutacc_torch

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  namespace py = pybind11;
  py::class_<
      kutacc_torch::ProcessGroupKutacc,
      c10d::Backend,
      c10::intrusive_ptr<kutacc_torch::ProcessGroupKutacc>>(module, "ProcessGroupKutacc")
      .def(
          "allgatherv_into_tensor",
          &kutacc_torch::ProcessGroupKutacc::allgatherv_into_tensor,
          py::arg("output"),
          py::arg("input"),
          py::arg("counts"),
          py::arg("displacements") = std::vector<int64_t>{});

  module.def(
      "create_backend",
      [](const c10::intrusive_ptr<c10d::Store>& store, int rank, int size, int64_t timeout_ms) {
        TORCH_CHECK(timeout_ms >= 0, "timeout must be non-negative");
        return c10::make_intrusive<kutacc_torch::ProcessGroupKutacc>(
            store, rank, size, std::chrono::milliseconds(timeout_ms));
      },
      py::arg("store"),
      py::arg("rank"),
      py::arg("size"),
      py::arg("timeout_ms"));
}
