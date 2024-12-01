#pragma once

#include "hetu/core/stream.h"
#include "hetu/common/macros.h"
#include "hetu/core/ndarray.h"
#include "hetu/graph/common.h"
#include "hetu/core/device.h"
#include "hetu/graph/tensor.h"
#include "hetu/graph/operator.h"
#include "hetu/utils/optional.h"
#include <stack>

namespace hetu {
namespace impl {

struct OpProfilerInfo {
  hetu::graph::OpType type;
  hetu::graph::OpName name;
  HTShapeList inputs_shape;
  double cost_time;
};

using ProfileId = uint64_t;

class Profile {
 public:
  Profile(bool enabled = true, bool use_cpu = false, bool use_cuda = false,
          bool record_shapes = false, bool profile_memory = false)
  : _id(_next_profile_id()), _enabled(enabled), _use_cpu(use_cpu), _use_cuda(use_cuda),
    _record_shapes(record_shapes), _profile_memory(profile_memory), _device(Device()) {}

  Profile(const Profile&) = delete;
  Profile& operator=(const Profile&) = delete;
  Profile(Profile&&) = delete;
  Profile& operator=(Profile&&) = delete;

  ~Profile() {
    Clear();
  }

  void Clear() {
    _op_record.clear();
    _ops.clear();
    _graph_view_record.clear();
  }

  void push(hetu::graph::OpType type, hetu::graph::OpName name,
            HTShapeList inputs_shape, double cost_time) {
    _op_record.push_back({type, name, inputs_shape, cost_time});
  }

  void push(hetu::graph::Operator& op) {
    HT_VALUE_ERROR_IF(!_enabled) << "The Profiler is not enabled";
    _ops.push_back(op);
  }

  void push(hetu::graph::OpType type, double total_time) {
    HT_VALUE_ERROR_IF(!_enabled) << "The Profiler is not enabled";
    _graph_view_record.push_back({type, total_time});
  }

  void set_device(Device device) {
    _device = device;
  }

  bool enabled() const {
    return _enabled;
  }

  bool record_shapes() const {
    return _record_shapes;
  }

  void sync_op() {
    if (!enabled())
      return;
    if (!_ops.empty()) {
      std::unordered_set<hetu::Stream> _sync_stream;
      for (auto& op : _ops) {
        _sync_stream.insert(op->instantiation_ctx().stream());
      }
      for (auto& stream : _sync_stream) {
        stream.Sync();
      }
      for (auto& op : _ops) {
        HTShapeList inputs_shape;
	      hetu::graph::Operator::for_each_input_tensor(op, [&](const hetu::graph::Tensor& input) {
          inputs_shape.push_back(input->shape());
        });
        _op_record.push_back({op->type(), op->name(), inputs_shape, op->TimeCost(0) * 1.0 / 1e6});
      }
      _ops.clear();
    }
    return;
  }

  std::vector<std::pair<hetu::graph::OpType, std::pair<double, std::pair<double, int>>>>
  get_optype_view() {
    sync_op();
    std::vector<std::pair<hetu::graph::OpType, std::pair<double, std::pair<double, int>>>> single_optype_total_time;
    std::map<hetu::graph::OpType, std::pair<double, int>> single_optype_total_time_unordered;

    for (auto& record: _op_record) {
      if (single_optype_total_time_unordered.find(record.type) == single_optype_total_time_unordered.end()) {
        single_optype_total_time_unordered[record.type] = {record.cost_time, 1};
      } else {
        single_optype_total_time_unordered[record.type].first += record.cost_time;
        single_optype_total_time_unordered[record.type].second++;
      }
    }
    for(auto& record : single_optype_total_time_unordered) {
      auto type = record.first;
      auto total_time = record.second.first;
      auto cnt = record.second.second;
      single_optype_total_time.push_back({type, {total_time, {total_time / cnt, cnt}}});
    }
    std::sort(single_optype_total_time.begin(), single_optype_total_time.end(),
      [&](std::pair<hetu::graph::OpType, std::pair<double, std::pair<double, int>>> x, std::pair<hetu::graph::OpType, std::pair<double, std::pair<double, int>>> y) {
                return x.second.second.first > y.second.second.first; });
    return single_optype_total_time;
  }

  std::vector<std::pair<std::pair<hetu::graph::OpType, HTShapeList>, std::pair<double, std::pair<double, int>>>>
  get_optype_with_inputs_view() {
    sync_op();

    std::vector<std::pair<std::pair<hetu::graph::OpType, HTShapeList>, std::pair<double, std::pair<double, int>>>> single_optype_with_inputs_total_time;
    std::map<std::pair<hetu::graph::OpType, HTShapeList>, std::pair< double, int>> single_optype_with_inputs_total_time_unordered;
    for (auto& record: _op_record) {
      if (single_optype_with_inputs_total_time_unordered.find({record.type, record.inputs_shape}) == single_optype_with_inputs_total_time_unordered.end()) {
        single_optype_with_inputs_total_time_unordered[{record.type, record.inputs_shape}] = {record.cost_time, 1};
      }
      else {
        single_optype_with_inputs_total_time_unordered[{record.type, record.inputs_shape}].first += record.cost_time;
        single_optype_with_inputs_total_time_unordered[{record.type, record.inputs_shape}].second++;
      }
    }
    for (auto& record : single_optype_with_inputs_total_time_unordered) {
      auto type = record.first.first;
      auto inputs_shape = record.first.second;
      auto total_time = record.second.first;
      auto cnt = record.second.second;
      single_optype_with_inputs_total_time.push_back({{type, inputs_shape}, {total_time, {total_time / cnt, cnt}}});
    }
    std::sort(single_optype_with_inputs_total_time.begin(), single_optype_with_inputs_total_time.end(),
      [&](std::pair<std::pair<hetu::graph::OpType, HTShapeList>, std::pair<double, std::pair<double, int>>> x, std::pair<std::pair<hetu::graph::OpType, HTShapeList>, std::pair<double, std::pair<double, int>>> y) {
            return x.second.second.first > y.second.second.first; });
    return single_optype_with_inputs_total_time;
  }

  std::vector<std::pair<hetu::graph::OpType, double>> get_graph_view() {
    return _graph_view_record;
  }

  std::vector<OpProfilerInfo> get_op_view() {
    sync_op();
    return  _op_record;
  }
 
 private:
  static ProfileId _next_profile_id();

 protected:
  ProfileId _id;
  bool _enabled;
  bool _record_shapes;
  bool _use_cpu;
  bool _use_cuda;
  bool _profile_memory;
  Device _device;
  std::vector<std::pair<std::string, double>> _graph_view_record;
  std::vector<OpProfilerInfo> _op_record;
  std::vector<hetu::graph::Operator> _ops;

  static void InitOnce() {
    std::call_once(Profile::_init_flag, Profile::Init);
  }

  static void Init();

 public:
  static Profile& make_new_profile(bool enabled = true, bool use_cpu = false,
                                   bool use_cuda = false, bool record_shapes = false,
                                   bool profile_memory = false) {
    InitOnce();
    auto res = std::make_shared<Profile>(enabled, use_cpu, use_cuda, record_shapes, profile_memory);
    Profile::_global_profile.push_back(res);
    return *Profile::_global_profile.back();
  }

  ProfileId id() {
    return _id;
  }

  static inline optional<std::shared_ptr<Profile>> get_cur_profile() {
    if (Profile::_cur_profile_ctx.empty() ||
        !Profile::_global_profile[Profile::_cur_profile_ctx.top()]->enabled())
      return std::nullopt;
    return Profile::_global_profile[Profile::_cur_profile_ctx.top()];
  }

  static inline std::shared_ptr<Profile> get_profile(ProfileId profile_id) {
    HT_VALUE_ERROR_IF(profile_id >= Profile::_global_profile.size())
      << "Profile with id " << profile_id << " does not exist";
    return Profile::_global_profile[profile_id];
  }

  static void push_profile_ctx(ProfileId id) {
    HT_VALUE_ERROR_IF(id >= Profile::_global_profile.size())
      << "Profile with id " << id << " does not exist";
    Profile::_cur_profile_ctx.push(id);
  }

  static void pop_profile_ctx() {
    Profile::_cur_profile_ctx.pop();
  }

 protected:
  static std::once_flag _init_flag;
  static std::vector<std::shared_ptr<Profile>> _global_profile;
  static thread_local std::stack<ProfileId> _cur_profile_ctx;
};

} // namespace impl
} // namespace hetu