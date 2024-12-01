#include "hetu/graph/headers.h"
#include "hetu/graph/ops/optimizer_update.h"
#include "hetu/graph/ops/Communication.h"
#include "hetu/impl/communication/nccl_comm_group.h"
#include "hetu/impl/communication/comm_group.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

bool OptimizerUpdateOpInterface::DoMapToParallelDevices(
  Operator& op, const DeviceGroup& placement_group) const {
  // use comm_op instead
  // if (placement_group.num_devices() > 1) {
  //   // TODO
  //   HT_NOT_IMPLEMENTED << "Fill this up with AllReduceOpImpl";
  // }
  return OpInterface::DoMapToParallelDevices(op, placement_group);
}

void SGDUpdateOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                                NDArrayList& outputs,
                                RuntimeContext& runtime_ctx) const {
  NDArray& param = outputs.at(0);
  const NDArray& grad = inputs.at(1);
  NDArray velocity;
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(),
                                  type(), hetu::impl::SGDUpdate, grad, param,
                                  velocity, learning_rate(), 0, false,
                                  op->instantiation_ctx().stream());
}

void SGDUpdateWithGradScalerOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                                              NDArrayList& outputs,
                                              RuntimeContext& runtime_ctx) const {
  NDArray& param = outputs.at(0);
  const NDArray& grad = inputs.at(1);
  const NDArray& infinite_count = inputs.at(2);
  NDArray velocity;
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(),
                               type(), hetu::impl::SGDUpdateWithGradScaler, grad, infinite_count, 
                               param, velocity, learning_rate(), 0, false,
                               op->instantiation_ctx().stream());
}

void MomentumUpdateOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                                     NDArrayList& outputs,
                                     RuntimeContext& runtime_ctx) const {
  NDArray& param = outputs.at(0);
  const NDArray& grad = inputs.at(1);
  NDArray velocity = inputs.at(2);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(),
                                  type(), hetu::impl::SGDUpdate, grad, param,
                                  velocity, learning_rate(), 0, false,
                                  op->instantiation_ctx().stream());
}

void AdamOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                           NDArrayList& outputs,
                           RuntimeContext& runtime_ctx) const {
  NDArray& param = outputs.at(0);
  const NDArray& grad = inputs.at(1);
  NDArray& mean = const_cast<NDArray&>(inputs.at(2));
  NDArray& variance = const_cast<NDArray&>(inputs.at(3));
  NDArray& step = const_cast<NDArray&>(inputs.at(4));
  if (!_multi_zero.at(op->graph().CUR_STRATEGY_ID)) {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(),
                                    type(), hetu::impl::Adam, grad, param,
                                    mean, variance, step, learning_rate(), 
                                    beta1(), beta2(), eps(), weight_decay(), 
                                    op->instantiation_ctx().stream());
  } else {
    // param is dup, should split as reduce-scatter
    // partial_grad -> reduce-scatter -> scatter_grad, use partial_grad distributed states to deduce scatter info (offset, index, comm_group...)
    HT_ASSERT(is_reduce_scatter_op(op->input(1)->producer()))
      << "Adam: zero input grad must be reduce-scatter result!"
      << ", grad producer = " << op->input(1)->producer()
      << ", grad = " << op->input(1)
      << ", param = " << op->input(0)
      << ", grad ds = " << op->input(1)->get_distributed_states().ds_info()
      << ", param ds = " << op->input(0)->get_distributed_states().ds_info();
    auto& reduce_scatter_op = op->input(1)->producer();
    auto& reduce_scatter_impl = reinterpret_cast<ReduceScatterOpImpl&>(reduce_scatter_op->body());
    auto& partial_grad = reduce_scatter_op->input(0);
    DeviceGroup comm_group = reduce_scatter_impl.comm_group();

    auto local_device_index = op->placement_group().get_index(op->placement());
    auto scatter_num = comm_group.num_devices();
    HT_ASSERT(scatter_num == partial_grad->get_distributed_states().get_dim(-2))
      << "Adam: comm_group num must equal to partial size!";
    auto param_size = param->numel();
    auto param_size_per_scatter = DIVUP(param_size, scatter_num); // todo: padding for reduce-scatter & all-gather
    const auto& param_ds = op->input(0)->get_distributed_states();
    auto scatter_index = partial_grad->get_distributed_states().map_device_to_state_index(local_device_index)[-2];
    auto param_start_index = param_size_per_scatter * scatter_index;
    auto param_end_index = param_start_index + param_size_per_scatter;
    HT_ASSERT(grad->numel() == param_size_per_scatter && param_end_index <= param_size) 
      << "now need param size can be div by dp group size! "
      << "got grad size = " << grad->numel() 
      << " vs. param_size_per_scatter = " 
      << param_size_per_scatter;

    auto param_scatter = NDArray(
      NDArrayMeta().set_shape(grad->shape())
                   .set_dtype(param->dtype())
                   .set_device(param->device()), 
      param->storage(), param->storage_offset() + param_start_index);
    // only update scatter part of param
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(),
                                    type(), hetu::impl::Adam, grad, param_scatter,
                                    mean, variance, step, learning_rate(), 
                                    beta1(), beta2(), eps(), weight_decay(), 
                                    op->instantiation_ctx().stream());
    // in-place allgather
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), 
                                    hetu::impl::AllGather, param_scatter, param, 
                                    comm_group, op->input(1)->get_distributed_states().get_split_dim(param_ds),
                                    op->instantiation_ctx().stream());
  }
}

// TODO: support zero
void AdamOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                const OpMeta& op_meta) const {
  const DistributedStates& ds_param = inputs.at(0)->get_distributed_states();
  const DistributedStates& ds_grad = inputs.at(1)->get_distributed_states();
  const DistributedStates& ds_mean = inputs.at(2)->get_distributed_states();
  const DistributedStates& ds_variance = inputs.at(3)->get_distributed_states();
  const DistributedStates& ds_step = inputs.at(4)->get_distributed_states();
  if (_multi_zero.at(Graph::GetGraph(Graph::cur_graph_ctx()).CUR_STRATEGY_ID)) {
    HT_ASSERT(ds_param.check_equal(ds_grad) && ds_mean.check_equal(ds_variance) && ds_param.check_equal(ds_mean))
      << "DistributedStates for param, grad, mean, variance should be equal!";
  } else {
    HT_ASSERT(ds_mean.check_equal(ds_variance) && ds_grad.check_equal(ds_mean))
      << "DistributedStates for grad, mean, variance should be equal for zero!";    
  }
  outputs.at(0)->set_distributed_states(ds_param);
}

Tensor MakeSGDUpdateOp(Tensor param, Tensor grad, float learning_rate,
                       OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<SGDUpdateOpImpl>(learning_rate),
                       {std::move(param), std::move(grad)}, std::move(op_meta))
    ->output(0);
}

Tensor MakeSGDUpdateWithGradScalerOp(Tensor param, Tensor grad, Tensor infinite_count, 
                                     float learning_rate, OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<SGDUpdateWithGradScalerOpImpl>(learning_rate),
                       {std::move(param), std::move(grad), std::move(infinite_count)}, std::move(op_meta))
    ->output(0);
}


Tensor MakeMomentumUpdateOp(Tensor param, Tensor grad, Tensor velocity,
                            float learning_rate, float momentum, bool nesterov,
                            OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<MomentumUpdateOpImpl>(
                         learning_rate, momentum, nesterov),
                       {std::move(param), std::move(grad), std::move(velocity)},
                       std::move(op_meta))
    ->output(0);
}

Tensor MakeAdamOp(Tensor param, Tensor grad, Tensor mean, Tensor variance,
                  float learning_rate, Tensor step, float beta1, float beta2, 
                  float eps, float weight_decay, OpMeta op_meta) {
  // pure tp needn't zero 
  std::vector<bool> multi_zero; 
  multi_zero.reserve(param->multi_distributed_states().size());   
  for (const auto& ds : param->multi_distributed_states()) {
    bool zero = (ds.get_dim(-1) > 1) && ds.zero();
    multi_zero.push_back(zero);
  }                 
  // HT_LOG_INFO << hetu::impl::comm::GetLocalDevice() << ": MakeAdamOp: param = " << param << ", multi zero = " << multi_zero;
  return Graph::MakeOp(std::make_shared<AdamOpImpl>(
                       learning_rate, multi_zero, beta1, beta2, eps, weight_decay),
                       {std::move(param), std::move(grad), std::move(mean), std::move(variance), std::move(step)},
                       std::move(op_meta))
    ->output(0);
}

} // namespace graph
} // namespace hetu
