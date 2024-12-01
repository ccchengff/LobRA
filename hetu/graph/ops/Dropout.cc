#include "hetu/graph/ops/Dropout.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"
#include "hetu/impl/random/CPURandomState.h"

namespace hetu {
namespace graph {

void DropoutOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                              NDArrayList& outputs,
                              RuntimeContext& ctx) const {
  uint64_t seed = hetu::impl::GenNextRandomSeed();
  // record seed for recomputed dropout in original op
  if (op->op_meta().is_recompute) {
    ctx.get_or_create(op->id()).put_uint64("seed", seed);
  }
  // get seed for recomputed dropout in recompute op
  if (op->op_meta().origin_op_id != -1) {
    seed = ctx.get(op->op_meta().origin_op_id).get_uint64("seed");
  }
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                               hetu::impl::Dropout, inputs.at(0), 1 - keep_prob(),
                               seed, outputs.at(0), outputs.at(1), op->instantiation_ctx().stream());
};

NDArrayList DropoutOpImpl::DoCompute(Operator& op,
                                     const NDArrayList& inputs,
                                     RuntimeContext& ctx) const {
  NDArrayList outputs;
  if (inplace()) {
    outputs = inputs;
    outputs.push_back(DoAllocOutput(op, inputs, 1, ctx));
  } else {
    outputs = DoAllocOutputs(op, inputs, ctx);
  }
  DoCompute(op, inputs, outputs, ctx);
  return outputs;
}

TensorList DropoutOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? MakeDropoutGradientOp(grad_outputs.at(0),
                                op->output(1), keep_prob(), inplace(),
                                op->grad_op_meta().set_name(op->grad_name()))
                               : Tensor()};
}

void DropoutGradientOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                                      NDArrayList& outputs,
                                      RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::DropoutGradient, inputs.at(0),
    inputs.at(1), 1 - keep_prob(), outputs.at(0), op->instantiation_ctx().stream());
};

NDArrayList DropoutGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                            RuntimeContext& ctx) const {
  NDArrayList outputs;
  if (fw_inplace()) {
    outputs.push_back(inputs[0]);
  } else {
    outputs.push_back(DoAllocOutput(op, inputs, 0, ctx));
  }
  DoCompute(op, inputs, outputs, ctx);
  return outputs;
}

Tensor MakeDropoutOp(Tensor input, double keep_prob,
                     OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<DropoutOpImpl>(keep_prob, false),
          {std::move(input)},
          std::move(op_meta))->output(0);
}

Tensor MakeDropoutInplaceOp(Tensor input, double keep_prob,
                            OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<DropoutOpImpl>(keep_prob, true),
          {std::move(input)},
          std::move(op_meta))->output(0);
}

Tensor MakeDropoutGradientOp(Tensor grad_output, Tensor mask,
                             double keep_prob, bool fw_inplace,
                             OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<DropoutGradientOpImpl>(keep_prob, fw_inplace),
          {std::move(grad_output), std::move(mask)},
          std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
