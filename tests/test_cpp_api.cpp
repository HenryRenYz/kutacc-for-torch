#include <kutacc_for_torch/process_group_kutacc.hpp>

int main() {
  auto* factory = &kutacc_torch::create_process_group_kutacc;
  auto* allgatherv = &kutacc_torch::allgatherv_into_tensor;
  return factory == nullptr || allgatherv == nullptr;
}

