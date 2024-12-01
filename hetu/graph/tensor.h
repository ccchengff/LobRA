#pragma once

#include "hetu/common/macros.h"
#include "hetu/core/ndarray.h"
#include "hetu/core/symbol.h"
#include "hetu/graph/common.h"
#include "hetu/graph/distributed_states.h"


namespace hetu {
namespace graph {

struct TensorIdentifier {
  GraphId graph_id;
  mutable OpId producer_id;
  int32_t output_id;
  TensorId tensor_id;
};

class TensorDef : public shared_ptr_target {
 protected:
  friend class OpDef;
  friend class Operator;
  friend class Tensor;
  friend class Graph;
  friend class ExecutableGraph;
  struct constrcutor_access_key {};

 public:
  TensorDef(const constrcutor_access_key&, TensorIdentifier ids,
            TensorName name, bool requires_grad, NDArrayMeta meta);

  ~TensorDef() = default;

  // disable copy constructor and move constructor
  TensorDef(const TensorDef&) = delete;
  TensorDef& operator=(const TensorDef&) = delete;
  TensorDef(TensorDef&&) = delete;
  TensorDef& operator=(TensorDef&&) = delete;

  TensorId id() const {
    return _ids.tensor_id;
  }

  const TensorName& name() const {
    return _name;
  }

  GraphId graph_id() const {
    return _ids.graph_id;
  }

  OpId producer_id() const {
    return _ids.producer_id;
  }

  // unsafe, may cause chaos of topo, so try to avoid using it
  void set_producer_id(const OpId& id) {
    _ids.producer_id = id;
  }

  const Graph& graph() const;

  Graph& graph();

  size_t num_strategy() const;

  size_t cur_strategy_id() const;
  
  const Operator& producer() const;

  Operator& producer();

  int32_t output_id() const noexcept {
    return _ids.output_id;
  }

  bool is_out_dep_linker() const noexcept {
    return output_id() < 0;
  }

  bool is_leaf() const {
    return is_variable();
  }

  bool is_variable() const;
  
  bool is_parameter() const;

  size_t num_consumers() const {
    return _consumers.size();
  }

  const OpRefList& consumers() const {
    return _consumers;
  }

  OpRefList& consumers() {
    return _consumers;
  }

  const Operator& consumer(size_t i) const;

  Operator& consumer(size_t i);

  const NDArrayMeta& meta() const noexcept {
    return _meta;
  }

  size_t ndim() const {
    return _meta.ndim();
  }

  size_t numel() const {
    return _meta.numel();
  }

  DataType dtype() const {
    return _meta.dtype;
  }

  const Device& device() const noexcept {
    return _meta.device;
  }

  bool is_cpu() const {
    return _meta.device.is_cpu();
  }

  bool is_cuda() const {
    return _meta.device.is_cuda();
  }

  const HTShape& shape() const {
    /*
    // 优先使用symbolic的value
    if (_symbolic) 
      return get_HTShape_from_SyShape(_symbolic_shape);
    */
    return _meta.shape;
  }

  int64_t shape(size_t axis) const {
    /*
    // 优先使用symbolic的value
    if (_symbolic) {
      auto shape = get_HTShape_from_SyShape(_symbolic_shape);
      return shape[axis];
    }
    */
    return _meta.shape[axis];
  }

  const HTStride& stride() const {
    /*
    // 优先使用symbolic的value
    if (_symbolic) {
      auto shape = get_HTShape_from_SyShape(_symbolic_shape);
      HTStride stride(shape.size());
      if (stride.size() > 0) {
        stride[stride.size() - 1] = 1;
        for (auto d = stride.size() - 1; d > 0; d--)
          stride[d - 1] = stride[d] * shape[d];
      }
      return stride;
    }
    */
    return _meta.stride;
  }

  int64_t stride(size_t axis) const {
    /*
    // 优先使用symbolic的value
    if (_symbolic) {
      auto shape = get_HTShape_from_SyShape(_symbolic_shape);
      HTStride stride(shape.size());
      if (stride.size() > 0) {
        stride[stride.size() - 1] = 1;
        for (auto d = stride.size() - 1; d > 0; d--)
          stride[d - 1] = stride[d] * shape[d];
      }
      return stride[axis];
    }
    */
    return _meta.stride[axis];
  }

  bool has_shape() const {
    return _meta.shape.size() > 0;
  }

  bool has_global_shape() const {
    return _global_shape.size() > 0;
  }

  const Device& placement() noexcept {
    return cur_distributed_states().get_placement();
  }

  void set_placement(const Device& p) {
    _meta.set_device(p);
    cur_distributed_states().set_placement(p);
  }

  const bool requires_grad() const noexcept {
    return _requires_grad;
  }

  void set_requires_grad(bool new_requires_grad) {
    _requires_grad = new_requires_grad;
  }

  const bool is_grad() const noexcept {
    return _is_grad;
  }

  void set_is_grad(bool is_grad) {
    _is_grad = is_grad;
  }

  bool is_contiguous() const {
    if (ndim() < 1 || numel() <= 1) { return true; }
    int64_t ndim_ = ndim();
    int64_t contiguous_stride = 1;
    for (int64_t i = ndim_ - 1; i >= 0; i--) {
      if (stride(i) != contiguous_stride)
        return false;
      contiguous_stride *= shape(i);
    }
    return true;
  }

  std::optional<OpId> get_contiguous_op_id() const {
    return _contiguous_op_id;
  }

  void set_contiguous_op_id(OpId contiguous_op_id) {
    _contiguous_op_id = std::make_optional<OpId>(contiguous_op_id);
  }

  NDArray get_or_compute();

  const DistributedStatesList& multi_distributed_states() const {
    return _distributed_states;
  }

  void set_multi_distributed_states(const DistributedStatesList& multi_ds) {
    _distributed_states = multi_ds;
  }

  bool check_multi_ds_equal(const DistributedStatesList& multi_ds) {
    HT_ASSERT(_distributed_states.size() == multi_ds.size())
      << "multi_ds size should equal the same as tensor has: " 
      << _distributed_states.size();
    for (size_t i = 0; i < _distributed_states.size(); i++) {
      if (!_distributed_states[i].check_equal(multi_ds[i])) {
        return false;
      }
    }
    return true;
  }

  DistributedStates& cur_distributed_states() {
    while (cur_strategy_id() >= _distributed_states.size()) {
      _distributed_states.push_back(DistributedStates());
    }
    return _distributed_states[cur_strategy_id()];
  }

  bool has_distributed_states() {
    return !cur_distributed_states().is_none();
  }

  const DistributedStates& get_distributed_states() {
    return cur_distributed_states();
  }

  void set_distributed_states(const DistributedStates& distributed_states) {
    cur_distributed_states().set_distributed_states(distributed_states);
  }

  const HTShape& global_shape() {
    if (has_global_shape()) {
      return _global_shape;
    }
    HTShape local_shape = shape();
    if (!has_distributed_states()) {
      return local_shape;
    }
    HTShape global_shape(local_shape.size());
    for (size_t d = 0; d < local_shape.size(); d++) {
      global_shape[d] = local_shape[d] * cur_distributed_states().get_dim(d);
    }
    _global_shape = global_shape;
    return _global_shape;
  }

  const DeviceGroup& placement_group() {
    return cur_distributed_states().get_placement_group();
  }

  void set_placement_group(const DeviceGroup& placement_group) {
    cur_distributed_states().set_placement_group(placement_group);
  }  

  bool symbolic() const {
    return _symbolic;
  }

  const SyShape& symbolic_shape() const {
    HT_ASSERT(_symbolic) << "symbolic_shape() can only work after calling copy/set/init symbolic_shape";
    return _symbolic_shape;
  }

  void set_shape(const HTShape& shape) {
    _meta.set_shape(shape);
    // no need to set _global_shape
    // since global_shape() method will automatically calc it
  }
  
  // copy constructor
  // don't know if it is a leaf
  void copy_symbolic_shape(const SyShape& symbolic_shape) {
    _symbolic = true;
    _symbolic_shape = symbolic_shape;
  }

  // leaf
  void set_symbolic_shape(const HTShape& shape) {
    _symbolic = true;
    set_HTShape_to_SyShape(shape, _symbolic_shape);
  }

  // leaf
  void init_symbolic_shape(const HTShape& shape) {
    _symbolic = true;
    for (auto x : _symbolic_shape) {
      x.reset();
    }
    set_HTShape_to_SyShape(shape, _symbolic_shape);
  }

  // leaf
  void init_symbolic_shape() {
    _symbolic = true;
    for (auto x : _symbolic_shape) {
      x.reset();
    }
    set_HTShape_to_SyShape(_meta.shape, _symbolic_shape);
  }

 protected:
  void AddConsumer(Operator& op);

  void DelConsumer(const Operator& op);

  // Walkaround methods to get the corresponding wrapper
  Tensor& get_self();

  const Tensor& get_self() const;
  
  const TensorIdentifier _ids;
  const TensorName _name;
  bool _requires_grad;
  NDArrayMeta _meta;
  OpRefList _consumers;
  bool _inform_graph_on_destruction;
  DistributedStatesList _distributed_states; // for multi ds deduce
  HTShape _global_shape;
  bool _is_grad{false};

  // Used when the tensor's shape is not fixed
  bool _symbolic;
  SyShape _symbolic_shape;

  std::optional<OpId> _contiguous_op_id{std::nullopt};
};

class Tensor : public shared_ptr_wrapper<TensorDef> {
 public:
  friend class Operator;
  friend class Graph;
  friend class ExecutableGraph;

  Tensor(TensorIdentifier ids, TensorName name, bool requires_grad,
         NDArrayMeta meta = {});

  Tensor() = default;

  ~Tensor();

 protected:
  size_t get_referrence_count() const {
    auto use_cnt_ = use_count();
    auto num_consumers_ = _ptr->num_consumers();
    HT_VALUE_ERROR_IF(use_cnt_ < (num_consumers_ + 1))
      << "Tensor " << _ptr->name() << " with " << num_consumers_ 
      << " consumers should have at least " << (num_consumers_ + 1) 
      << " use counts, but got " << use_cnt_;
    return use_cnt_ - (num_consumers_ + 1);
  }
  
  /******************************************************
   * Helper functions
   ******************************************************/ 
 public: 
  template <typename UnaryFunction>
  static void for_each_consumer(Tensor& tensor, UnaryFunction fn) {
    for (auto& op : tensor->_consumers)
      fn(op.get());
  }

  template <typename UnaryFunction>
  static void for_each_consumer(const Tensor& tensor, UnaryFunction fn) {
    for (const auto& op : tensor->_consumers)
      fn(op.get());
  }

  template <typename UnaryPredicate>
  static bool all_consumers_of(Tensor& tensor, UnaryPredicate pred) {
    for (auto& op : tensor->_consumers)
      if (!pred(op))
        return false;
    return true;
  }

  template <typename UnaryPredicate>
  static bool all_consumers_of(const Tensor& tensor, UnaryPredicate pred) {
    for (const auto& op : tensor->_consumers)
      if (!pred(op))
        return false;
    return true;
  }

  template <typename UnaryPredicate>
  static bool any_consumer_of(Tensor& tensor, UnaryPredicate pred) {
    for (auto& op : tensor->_consumers)
      if (pred(op))
        return true;
    return false;
  }

  template <typename UnaryPredicate>
  static bool any_consumer_of(const Tensor& tensor, UnaryPredicate pred) {
    for (const auto& op : tensor->_consumers)
      if (pred(op))
        return true;
    return false;
  }
};

std::ostream& operator<<(std::ostream&, const Tensor&);

} // namespace graph
} // namespace hetu
