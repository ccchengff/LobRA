#pragma once

#include "hetu/graph/common.h"
#include <functional>

namespace hetu {
namespace graph {

class DistributedStates;
using DistributedStatesList = std::vector<DistributedStates>;

class DistributedStates {
 public:
  DistributedStates() : _device_num(-1), _states({}), _order({}), _zero(false) {};
  DistributedStates(DeviceGroup placement_group, std::unordered_map<int32_t, int32_t> states, 
                    std::vector<int32_t> order = {}, bool zero = false) {
    _placement_group = placement_group;
    _device_num = placement_group.num_devices();
    _zero = zero;
    set_states(states);
    set_order(order); 
  }
  // independent distributed states, the placement_group should be assigned when binding with tensor
  DistributedStates(int32_t device_num, std::unordered_map<int32_t, int32_t> states, 
                    std::vector<int32_t> order = {}, bool zero = false) {
    _placement_group = DeviceGroup();
    _device_num = device_num;
    _zero = zero;
    set_states(states);
    set_order(order); 
  }
  DistributedStates(const DistributedStates& ds) {
    _device_num = -1;
    if (ds._device_num == -1) return; // do nothing if ds is invalid
    set_distributed_states(ds);
  }

  void set_placement_group(const DeviceGroup& placement_group);

  void set_placement(const Device& placement);

  void set_distributed_states(const DistributedStates& dst_distributed_states);

  bool is_none() const;
  
  bool is_valid() const;

  const std::unordered_map<int32_t, int32_t>& get_states() const {
    return _states;
  }

  int32_t states(int32_t key) const {
    return get_dim(key);
  } 

  const std::vector<int32_t>& get_order() const {
    return _order;
  }

  int32_t order(int32_t i) const {
    return _order.at(i);
  }

  bool has_placement_group() const {
    return !_placement_group.empty();
  }
  
  bool zero() const {
    return _zero;
  }

  void set_zero(bool zero) {
    _zero = zero;
  }

  const DeviceGroup& get_placement_group() const {
    return _placement_group;
  }

  const Device& get_placement() const {
    return _placement;
  }

  int32_t get_placement_index() const {
    return _placement_group.get_index(_placement);
  }

  int32_t get_device_num() const {
    return _device_num;
  }

  int32_t get_split_num() const {
    int32_t split_num = 1;
    for (auto o : _order) {
      if (o >= 0) {
        split_num *= states(0);
      }
    }
    return split_num;
  }

  std::unordered_map<int32_t, int32_t> combine_states(const std::pair<std::vector<int32_t>, int32_t>& src2dst) const;
  static std::unordered_map<int32_t, int32_t> combine_states(const std::pair<std::vector<int32_t>, int32_t>& src2dst, 
                                                             const std::unordered_map<int32_t, int32_t>& ori_states);
  std::vector<int32_t> combine_order(const std::pair<std::vector<int32_t>, int32_t>& src2dst) const;
  static std::vector<int32_t> combine_order(const std::pair<std::vector<int32_t>, int32_t>& src2dst, 
                                            const std::vector<int32_t>& ori_order);
  bool equal_states_and_order(const std::unordered_map<int32_t, int32_t>& states1, const std::vector<int32_t>& order1,
                              const std::unordered_map<int32_t, int32_t>& states2, const std::vector<int32_t>& order2) const;
  bool check_equal(const DistributedStates& dst_distributed_states) const;
  bool check_max_dim(int32_t max_dim) const;
  bool check_pure_duplicate() const;    
  bool check_combine(const DistributedStates& dst_distributed_states, const std::pair<std::vector<int32_t>, int32_t>& src2dst) const;

  std::unordered_map<int32_t, int32_t> reduce_states(int dim) const;
  std::vector<int32_t> reduce_order(int dim) const;
  bool check_reduce_dim(const DistributedStates& dst_distributed_states, int dim) const;

  bool check_split(const DistributedStates& dst_distributed_states) const;
  bool check_scatter(const DistributedStates& dst_distributed_states) const;
  bool check_allreduce(const DistributedStates& dst_distributed_states) const;
  bool check_allgather(const DistributedStates& dst_distributed_states) const;
  bool check_reducescatter(const DistributedStates& dst_distributed_states) const;
  bool check_boradcast(const DistributedStates& dst_distributed_states) const;
  bool check_reduce(const DistributedStates& dst_distributed_states) const;  

  int32_t get_dim(int32_t index) const;
  int32_t get_split_dim(const DistributedStates& merged_distributed_states) const;
  std::vector<int32_t> get_loop_sizes() const;
  std::unordered_map<int32_t, int32_t> map_device_to_state_index(int32_t device_index) const;
  int32_t get_dup_group_index(int32_t device_index) const;
  DeviceGroup get_devices_by_dim(int32_t dim, int32_t local_device_idx, DeviceGroup group) const;
  std::string ds_info() const;

 protected:
  // should call set_states and set_order at the same time
  void set_states(const std::unordered_map<int32_t, int32_t>& states);

  void set_order(const std::vector<int32_t>& order);

  bool _zero;
  std::unordered_map<int32_t, int32_t> _states; // {dimension: split_num}, {-2: partial, -1: duplicate, 0~n-1: dimension}
  std::vector<int32_t> _order; // for device mapping
  DeviceGroup _placement_group; // must be assigned when binding with tensor
  int32_t _device_num; // if placement_group is null, the device_num must be assigned
  Device _placement;
};

} // namespace autograd
} // namespace hetu