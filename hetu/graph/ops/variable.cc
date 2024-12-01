#include "hetu/graph/ops/variable.h"
#include "hetu/graph/headers.h"

namespace hetu {
namespace graph {

NDArrayList VariableOpImpl::DoAllocOutputs(Operator& op,
                                           const NDArrayList& inputs,
                                           RuntimeContext& runtime_ctx) const {
  if (_init != nullptr) {
    Graph::AllocVariableData(op->output(0), *_init);
  } else {
    if (_copy_provided_data || dtype() != _provided_data->dtype() ||
        op->instantiation_ctx().placement != _provided_data->device()) {
      Graph::AllocVariableData(op->output(0),
                               ProvidedInitializer(_provided_data));
    } else {
      Graph::RegisterVariableData(op->output(0), _provided_data);
    }
  }
  return {Graph::GetVariableData(op->output(0))};
}

NDArrayList ParallelVariableOpImpl::DoAllocOutputs(Operator& op,
                                                   const NDArrayList& inputs,
                                                   RuntimeContext& runtime_ctx) const {
  auto ds = _multi_ds[op->graph().CUR_STRATEGY_ID];
  auto local_idx = _local_idx.empty() ? -1 : _local_idx[op->graph().CUR_STRATEGY_ID];
  HT_ASSERT(_init == nullptr || local_idx != -1)
    << "ParallelVariableOp: when use initializer, local_idx "
    << "must be assigned when local_device is in pipeline device_group!";

  if (_init != nullptr) {
    int32_t dup_group_idx = ds.get_dup_group_index(local_idx);
    // support 100 different duplicate group to set different seed
    uint64_t seed = 2023 + op->id() * 100 + dup_group_idx;
    HT_LOG_DEBUG << hetu::impl::comm::GetLocalDevice() << ": " << op << " inits by initializer.";
    // TODO: reset variable data also need parallel version
    Graph::AllocVariableData(op->output(0), *_init, seed, _global_shape);
  } else {
    auto& provided_data = _multi_provided_data.empty() ? 
      _provided_data : _multi_provided_data[op->graph().CUR_STRATEGY_ID]; 
    if (_copy_provided_data || dtype() != provided_data->dtype() ||
        op->instantiation_ctx().placement != provided_data->device()) {
      HT_LOG_DEBUG << hetu::impl::comm::GetLocalDevice() << ": " << op << " inits by provided data.";
      Graph::AllocVariableData(op->output(0),
                               ProvidedInitializer(provided_data));
    } else {
      Graph::RegisterVariableData(op->output(0), provided_data);
    }
  }
  return {Graph::GetVariableData(op->output(0))};
}

// in_degree=0, should set distributed states manually
Tensor MakeVariableOp(const Initializer& init, HTShape shape, 
                      DataType dtype, bool requires_grad, 
                      const DistributedStates& ds, OpMeta op_meta) {
  auto out = Graph::MakeOp(std::make_shared<VariableOpImpl>(
                           init, std::move(shape), dtype, requires_grad), TensorList(), 
                           std::move(op_meta.set_is_deduce_states(false)))->output(0);
  if (!ds.is_none()) {
    HT_ASSERT(ds.is_valid() && ds.get_dim(-2) == 1)
      << "DistributedStates for VariableOp must be valid! got: " 
      << ds.ds_info();
    out->set_distributed_states(ds);
  }
  return out;
}

// in_degree=0, should set distributed states manually
Tensor MakeVariableOp(NDArray provided_data, bool copy_provided_data, 
                      DataType dtype, bool requires_grad, 
                      const DistributedStates& ds, OpMeta op_meta) {
  auto out = Graph::MakeOp(std::make_shared<VariableOpImpl>(
                           std::move(provided_data), copy_provided_data, dtype,
                           requires_grad), TensorList(), 
                           std::move(op_meta.set_is_deduce_states(false)))->output(0);
  if (!ds.is_none()) {
    HT_ASSERT(ds.is_valid() && ds.get_dim(-2) == 1)
      << "DistributedStates for VariableOp must be valid! got: " 
      << ds.ds_info();
    out->set_distributed_states(ds);
  }
  return out;
}

Tensor MakeParameterOp(const Initializer& init, HTShape shape, 
                       DataType dtype, bool requires_grad, 
                       const DistributedStates& ds, OpMeta op_meta) {
  auto out = MakeVariableOp(init, std::move(shape), dtype, 
                            requires_grad, ds, std::move(op_meta));
  Graph::MarkAsParameter(out);
  return out;
}

Tensor MakeParameterOp(NDArray provided_data, bool copy_provided_data, 
                       DataType dtype, bool requires_grad, 
                       const DistributedStates& ds, OpMeta op_meta) {
  auto out = MakeVariableOp(std::move(provided_data), copy_provided_data, 
                            dtype, requires_grad, ds, std::move(op_meta));
  Graph::MarkAsParameter(out);
  return out;
}

Tensor MakeParallelVariableOp(const Initializer& init, HTShape global_shape, 
                              const DistributedStatesList& multi_ds, std::vector<int64_t> local_idx,
                              DataType dtype, bool requires_grad, OpMeta op_meta) {
  // init local_idx vector
  if (local_idx.size() == 1) {
    local_idx.resize(multi_ds.size(), local_idx[0]);
  } else {
    HT_ASSERT(local_idx.size() == multi_ds.size()) 
      << "ParallelVariableOp: local_idx size must equal to multi_ds!";
  }
  auto out = Graph::MakeOp(std::make_shared<ParallelVariableOpImpl>(
                           init, std::move(global_shape), multi_ds, 
                           std::move(local_idx), dtype, requires_grad),
                           TensorList(), std::move(op_meta.set_is_deduce_states(false)))->output(0);
  // assign multi ds for variable
  auto& graph = out->graph();
  for (size_t cur_strategy_id = 0; cur_strategy_id < graph.NUM_STRATEGY; cur_strategy_id++) {
    const auto& ds = multi_ds[cur_strategy_id];
    HT_ASSERT(ds.is_valid() && ds.get_dim(-2) == 1)
      << "DistributedStates for ParallelVariableOp must be valid! got: " 
      << ds.ds_info();
    graph.CUR_STRATEGY_ID = cur_strategy_id;
    out->set_distributed_states(ds);
  }
  graph.CUR_STRATEGY_ID = 0;
  return out;
}

Tensor MakeParallelVariableOp(NDArray provided_data, 
                              const DistributedStatesList& multi_ds, 
                              bool copy_provided_data, DataType dtype, 
                              bool requires_grad, OpMeta op_meta) {
  // auto placement_group = op_meta.device_group;
  auto out = Graph::MakeOp(std::make_shared<ParallelVariableOpImpl>(
                           provided_data, copy_provided_data, 
                           multi_ds, dtype, requires_grad),
                           TensorList(), std::move(op_meta.set_is_deduce_states(false)))->output(0);
  // assign multi ds for variable
  auto& graph = out->graph();
  for (size_t cur_strategy_id = 0; cur_strategy_id < graph.NUM_STRATEGY; cur_strategy_id++) {
    auto& ds = multi_ds[cur_strategy_id];
    HT_ASSERT(ds.is_valid() && ds.get_dim(-2) == 1)
      << "DistributedStates for ParallelVariableOp must be valid! got: " 
      << ds.ds_info();
    graph.CUR_STRATEGY_ID = cur_strategy_id;
    out->set_distributed_states(ds);
  }
  graph.CUR_STRATEGY_ID = 0;
  return out;  
}

Tensor MakeParallelVariableOp(NDArrayList multi_provided_data, 
                              DistributedStatesList multi_ds, 
                              bool copy_provided_data, DataType dtype, 
                              bool requires_grad, OpMeta op_meta) {
  auto out = Graph::MakeOp(std::make_shared<ParallelVariableOpImpl>(
                           std::move(multi_provided_data), copy_provided_data, 
                           std::move(multi_ds), dtype, requires_grad),
                           TensorList(), std::move(op_meta.set_is_deduce_states(false)))->output(0);
  // assign multi ds for variable
  auto& graph = out->graph();
  for (size_t cur_strategy_id = 0; cur_strategy_id < graph.NUM_STRATEGY; cur_strategy_id++) {
    auto& ds = multi_ds[cur_strategy_id];
    HT_ASSERT(ds.is_valid() && ds.get_dim(-2) == 1)
      << "DistributedStates for ParallelVariableOp must be valid! got: " 
      << ds.ds_info();
    graph.CUR_STRATEGY_ID = cur_strategy_id;
    out->set_distributed_states(ds);
  }
  graph.CUR_STRATEGY_ID = 0;
  return out;  
}

Tensor MakeParallelParameterOp(const Initializer& init, HTShape global_shape, 
                               const DistributedStatesList& multi_ds, std::vector<int64_t> local_idx,
                               DataType dtype, bool requires_grad, OpMeta op_meta) {
  auto out = MakeParallelVariableOp(init, std::move(global_shape), multi_ds, std::move(local_idx), 
                                    dtype, requires_grad, std::move(op_meta));
  Graph::MarkAsParameter(out);
  return out;                                    
}

Tensor MakeParallelParameterOp(NDArray provided_data, 
                               const DistributedStatesList& multi_ds, 
                               bool copy_provided_data, DataType dtype, 
                               bool requires_grad, OpMeta op_meta) {    
  auto out = MakeParallelVariableOp(std::move(provided_data), multi_ds, copy_provided_data, 
                                    dtype, requires_grad, std::move(op_meta));
  Graph::MarkAsParameter(out);
  return out;
}

Tensor MakeParallelParameterOp(NDArrayList multi_provided_data, 
                               DistributedStatesList multi_ds, 
                               bool copy_provided_data, DataType dtype, 
                               bool requires_grad, OpMeta op_meta) {    
  auto out = MakeParallelVariableOp(std::move(multi_provided_data), std::move(multi_ds), 
                                    copy_provided_data, dtype, requires_grad, std::move(op_meta));
  Graph::MarkAsParameter(out);
  return out;
}

} // namespace graph
} // namespace hetu
