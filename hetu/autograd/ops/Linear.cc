#include "hetu/autograd/ops/MatMul.h"
#include "hetu/autograd/ops/Linear.h"
#include "hetu/autograd/ops/ReduceSum.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void LinearOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                            RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(), hetu::impl::Linear,
                               inputs.at(0), trans_a(), inputs.at(1), trans_b(),
                               inputs.at(2), outputs.at(0), stream());
}

TensorList LinearOpDef::DoGradient(const TensorList& grad_outputs) {
  const Tensor& grad_c = grad_outputs.at(0);
  Tensor& a = _inputs[0];
  Tensor& b = _inputs[1];
  Tensor grad_a;
  Tensor grad_b;
  auto g_op_meta = grad_op_meta();
  if (!trans_a() && !trans_b()) {
    // case 1: c = Linear(a, b)
    // grad_a = Linear(grad_c, b^T), grad_b = Linear(a^T, grad_c)
    grad_a = MatMulOp(grad_c, b, false, true, g_op_meta.set_name(grad_name(0)))
               ->output(0);
    grad_b = MatMulOp(a, grad_c, true, false, g_op_meta.set_name(grad_name(1)))
               ->output(0);
  } else if (trans_a() && !trans_b()) {
    // case 2: c = Linear(a^T, b)
    // grad_a = Linear(b, grad_c^T), grad_b = Linear(a, grad_c)
    grad_a = MatMulOp(b, grad_c, false, true, g_op_meta.set_name(grad_name(0)))
               ->output(0);
    grad_b = MatMulOp(a, grad_c, false, false, g_op_meta.set_name(grad_name(1)))
               ->output(0);
  } else if (!trans_a() && trans_b()) {
    // case 3: c = Linear(a, b^T)
    // grad_a = Linear(grad_c, b), grad_b = Linear(grad_c^T, a)
    grad_a = MatMulOp(grad_c, b, false, false, g_op_meta.set_name(grad_name(0)))
               ->output(0);
    grad_b = MatMulOp(grad_c, a, true, false, g_op_meta.set_name(grad_name(1)))
               ->output(0);
  } else {
    // case 4: c = Linear(a^T, b^T)
    // grad_a = Linear(b^T, grad_c^T), grad_b = Linear(grad_c^T, a^T)
    grad_a = MatMulOp(b, grad_c, true, true, g_op_meta.set_name(grad_name(0)))
               ->output(0);
    grad_b = MatMulOp(grad_c, a, true, true, g_op_meta.set_name(grad_name(1)))
               ->output(0);
  }
  Tensor grad_bias = ReduceSumOp(grad_outputs.at(0), {0}, {false},
                                 g_op_meta.set_name(grad_name(2)))
                       ->output(0);
  return {grad_a, grad_b, grad_bias};
}

void LinearOpDef::DoInferMeta() {
  auto a = _inputs[0];
  auto b = _inputs[1];
  if (a->has_shape() && b->has_shape()) {
    HT_ASSERT(a->ndim() == 2 && b->ndim() == 2)
      << "Failed to construct the \"" << type() << "\" operation "
      << "(with name \"" << name() << "\"): "
      << "Dimensions must be 2. "
      << "Got " << a->ndim() << ", " << b->ndim() << ".";
    int64_t dim_a = a->shape(trans_a() ? 0 : 1);
    int64_t dim_b = b->shape(trans_b() ? 1 : 0);
    HT_ASSERT(dim_a == -1 || dim_b == -1 || dim_a == dim_b)
      << "Failed to construct the \"" << type() << "\" operation "
      << "(with name \"" << name() << "\"): "
      << "Dimensions must be compatible. "
      << "Got " << dim_a << " vs. " << dim_b << ". "
      << "Input shapes: " << a->shape() << " vs. " << b->shape() << ".";
  }
  HTShape shape = {-1, -1};
  if (a->has_shape())
    shape[0] = a->shape(trans_a() ? 1 : 0);
  if (b->has_shape())
    shape[1] = b->shape(trans_b() ? 0 : 1);
  HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape).set_device(_inputs[0]->device()));
}

HTShapeList LinearOpDef::DoInferShape(const HTShapeList& input_shapes) {
  const HTShape& a = input_shapes.at(0);
  const HTShape& b = input_shapes.at(1);
  HT_ASSERT(a.size() == 2 && b.size() == 2 &&
            a.at(trans_a() ? 0 : 1) == b.at(trans_b() ? 1 : 0))
    << "Invalid input shapes for " << type() << ":"
    << " (shape_a) " << a << " (shape_b) " << b << " (transpose_a) "
    << trans_a() << " (transpose_b) " << trans_b();
  return {{a.at(trans_a() ? 1 : 0), b.at(trans_b() ? 0 : 1)}};
}

void LinearOpDef::DoDeduceStates() {
  Tensor& a = _inputs[0];
  Tensor& b = _inputs[1];
  Tensor& bias = _inputs[2];
  DistributedStates ds_a = a->get_distributed_states();
  DistributedStates ds_b = b->get_distributed_states();
  DistributedStates ds_bias = bias->get_distributed_states();
  int32_t device_num = ds_a.get_device_num();

  HT_ASSERT(ds_a.is_valid() && ds_b.is_valid() && ds_bias.is_valid() 
            && ds_a.get_device_num() == ds_b.get_device_num()
            && ds_b.get_device_num() == ds_bias.get_device_num())
            << "cannot convert src distributed states to unpaired dst distributed states!";
  // check bias states
  if (trans_b()) { // bias shape = (b.shape[0], )
    HT_ASSERT(ds_b.get_dim(0) == ds_bias.get_dim(0))
      << "LinearOp: bias should split same with dimension 0 of b";
  } else { // bias shape = (b.shape[1], )
    HT_ASSERT(ds_b.get_dim(1) == ds_bias.get_dim(0))
      << "LinearOp: bias should split same with dimension 1 of b";
  }          
  // l,r to result states map  
  std::vector<std::unordered_map<int32_t, int32_t>> l2res_case({
    {{-1, 1}, {0, 0}, {1, -2}}, // no trans
    {{-1, 1}, {1, 0}, {0, -2}}  // trans A
  });
  auto& l2res_map = l2res_case[trans_a()];
  std::vector<std::unordered_map<int32_t, int32_t>> r2res_case({
    {{-1, 0}, {0, -2}, {1, 1}}, // no trans
    {{-1, 0}, {0, 1}, {1, -2}}  // trans B
  });
  auto& r2res_map = r2res_case[trans_b()];
  // deduce states
  int32_t lrow = ds_a.get_dim(trans_a());
  int32_t lcol = ds_a.get_dim(1-trans_a());
  int32_t rrow = ds_b.get_dim(trans_b());
  int32_t rcol = ds_b.get_dim(1-trans_b());
  HT_ASSERT(lcol == rrow) << "Linear: tensor a.dimension[1] " << lcol 
    << " must be equal to tensor b.dimension[0] " << rrow;
  // if output states contains partial, then requires bias also should be partial
  HT_ASSERT(lcol == ds_bias.get_dim(-2))
    << "Linear: partial in output states = " << lcol << " should be equal to partial of bias = " << ds_bias.get_dim(-2);
  std::unordered_map<int32_t, int32_t> res_states({
    {-2, lcol}, {-1, device_num/(lcol*lrow*rcol)}, {0, lrow}, {1, rcol}
  });
  // deduce order
  std::vector<int32_t> lorder = ds_a.get_order();
  std::vector<int32_t> rorder = ds_b.get_order();
  auto get_new_order = [](std::unordered_map<int32_t, int32_t>& _map,
  std::vector<int32_t>& _order) -> std::vector<int32_t> {
    std::vector<int32_t> new_order;
    for (int32_t x : _order) {
      new_order.push_back(_map[x]);
    }
    return new_order;
  };
  auto get_index = [](std::vector<int32_t>& _order, int32_t val) -> int32_t {
    auto it = std::find(_order.begin(), _order.end(), val);
    HT_ASSERT(it != _order.end()) << "dimension " << val << " is not in order!";
    return it - _order.begin();
  };
  auto new_lorder = get_new_order(l2res_map, lorder);
  auto new_rorder = get_new_order(r2res_map, rorder);
  if (new_lorder != new_rorder) {
    new_lorder[get_index(new_lorder, 1)] = -1;
    new_rorder[get_index(new_rorder, 0)] = -1;
    HT_ASSERT(new_lorder == new_rorder) << "new_lorder is not equal to new_rorder!";
  } else if (std::find(new_lorder.begin(), new_lorder.end(), 0) != new_lorder.end()
             && ds_a.get_dim(-1) > 1) {
    int32_t ind0 = get_index(new_lorder, 0);
    int32_t ind1 = get_index(new_lorder, 1);
    if (ind0 > ind1) {
      int32_t tmp = ind0;
      ind0 = ind1;
      ind1 = tmp;
    }
    HT_ASSERT(ind0 + 1 == ind1) << "ind0 + 1 != ind1";
    new_lorder.insert(new_lorder.begin() + ind1, -1);
  }
  std::vector<int32_t> res_order(new_lorder);
  // set distributed states for result c
  Tensor& c = _outputs[0];
  c->set_distributed_states({device_num, res_states, res_order});
}

} // namespace autograd
} // namespace hetu
