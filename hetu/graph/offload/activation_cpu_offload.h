#pragma once

#include "hetu/graph/common.h"
#include "hetu/graph/executable_graph.h"
#include "hetu/graph/graph.h"
#include <functional>

namespace hetu {
namespace graph {

class ActivationCPUOffload {
 public:
  static bool enabled() {
    return _enabled;
  }

  static void set_cpu_offload_enabled() {
    _enabled = true;
  }

  static void reset_cpu_offload_enabled() {
    _enabled = false;
  }

  static void OffloadToCPU(const OpRefList& topo_order);

 protected:

  static bool IsNoOffloadOp(Operator& op);

  static void OffloadTensorToCPU(const OpRefList& topo_order, const Tensor& tensor);

  static bool _enabled;
};

} // namespace graph
} // namespace hetu