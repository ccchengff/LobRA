#pragma once

#include "hetu/graph/common.h"
#include "hetu/graph/executable_graph.h"
#include "hetu/graph/graph.h"
#include <functional>

namespace hetu {
namespace graph {

class Recompute {
 public:
  static bool enabled() {
    return _enabled;
  }

  static void set_recompute_enabled() {
    _enabled = true;
  }

  static void reset_recompute_enabled() {
    _enabled = false;
  }

  static void InsertRecomputedOps(const OpRefList& topo_order);

 protected:
  static bool IsNoRecomputedOp(Operator& op);

  static void GetMaxRecomputeSubGraph(Op2OpRefMap& recompute_subgraph, bool get_inputs, bool get_outputs);

  static bool HasFilterOpInPath(const Operator& op, std::function<bool(const Operator &)> filter_fn,
                                std::unordered_map<OpId, bool>& has_filter_op_map);

  static Operator& DuplicateRecomputedOp(const Operator& origin_op, const Op2OpRefMap& filtered_recomputed_ops,
                                         const TensorList& first_mapped_grad_inputs, Op2OpMap& origin_to_recomputed_map,
                                         ExecutableGraph& cur_exec_graph);

  static bool _enabled;
};

} // namespace graph
} // namespace hetu