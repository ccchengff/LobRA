#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class CheckFiniteOpImpl;
class CheckFiniteOp;

class CheckFiniteOpImpl final : public OpInterface {
 private:
  friend class CheckFiniteOp;
  struct constrcutor_access_key {};

 public:
  CheckFiniteOpImpl()
  : OpInterface(quote(CheckFiniteOp)) {
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    NDArrayMeta out_meta = inputs[0]->meta();
    out_meta.set_shape({1}).set_dtype(DataType::FLOAT32);
    return {out_meta};
  };

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    return OpInterface::operator==(rhs); 
  }
};

Tensor MakeCheckFiniteOp(Tensor input, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
