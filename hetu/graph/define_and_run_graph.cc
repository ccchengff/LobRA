#include "hetu/graph/define_and_run_graph.h"
#include "hetu/graph/executable_graph.h"
#include "hetu/graph/switch_exec_graph.h"
#include "hetu/graph/ops/variable.h"
#include "hetu/graph/ops/optimizer_update.h"
#include "hetu/graph/autocast/autocast.h"
#include "hetu/impl/communication/comm_group.h"
#include "hetu/impl/communication/mpi_comm_group.h"
#include "hetu/impl/communication/nccl_comm_group.h"
#include "hetu/graph/recompute/recompute.h"
#include "hetu/graph/offload/activation_cpu_offload.h"
#include "hetu/impl/memory/CUDACachingMemoryPool.cuh"

namespace hetu {
namespace graph {

// changing parallel plan
static size_t change_parallel_test_case = 0;

Operator& DefineAndRunGraph::MakeOpInner(std::shared_ptr<OpInterface> body,
                                         TensorList inputs, OpMeta op_meta) {
  _check_all_inputs_in_graph(inputs, op_meta.extra_deps);
  // for optimization passes
  op_meta = op_meta.set_is_recompute(Recompute::enabled())
                   .set_is_cpu_offload(ActivationCPUOffload::enabled());
  auto& op = MakeAndAddOp(std::move(body), std::move(inputs), std::move(op_meta));
  // record the ops that have an explicit device group setting
  // which will then used to deduce the pp stages
  if (op->device_groups().size() == NUM_STRATEGY) {
    _ops_with_device_groups.emplace_back(op);
  } 
  return _op_indexing[op->id()];
}

void DefineAndRunGraph::ResetVariableDataInner(const Tensor& tensor,
                                               const Initializer& init) {
  // Mark an add-on initializer.
  _add_on_inits[tensor->id()] = std::unique_ptr<Initializer>(init.copy());
  if (_is_active) {
    auto& tensor_to_exec_tensor_mapping = _exec_graph_plan_pool[_active_exec_plan].tensor_to_exec_tensor_mapping;
    auto it = tensor_to_exec_tensor_mapping.find(tensor->id());
    if (it != tensor_to_exec_tensor_mapping.end()) {
      // The op has been instantiated in the current active graph. Also let the executable graph reset it.
      Graph::ResetVariableData(it->second, init);
    }
  }
}

NDArray DefineAndRunGraph::GetDetachedVariableDataInner(const Tensor& tensor) {
  if (_is_active) {
    auto& tensor_to_exec_tensor_mapping = _exec_graph_plan_pool[_active_exec_plan].tensor_to_exec_tensor_mapping;
    auto it_1 = tensor_to_exec_tensor_mapping.find(tensor->id());
    if (it_1 == tensor_to_exec_tensor_mapping.end()) {
      // The tensor is not in current active exec graph.
      // Question: store the data on different devices? For now, store all on CPU and return.
      auto ret = NDArray::empty(tensor->shape(), Device(kCPU), tensor->dtype(), kBlockingStream);
      auto it_2 = _add_on_inits.find(tensor->id());
      // Note _add_on_inits has a higher priority than the original tensor initializer.
      if (it_2 != _add_on_inits.end()) {
        HT_LOG_TRACE << "The data is reset, but not in current active exec graph, "
          << "so we get the data of the variable from the DefineAndRun graph.";
        it_2->second->Init(ret);
      } else {
        HT_LOG_TRACE << "The data is not in current active exec graph, " 
          << "so we get the data of the variable from its initializer.";
        if (tensor->has_distributed_states())
          dynamic_cast<ParallelVariableOpImpl&>(tensor->producer()->body()).initializer().Init(ret);
        else
          dynamic_cast<VariableOpImpl&>(tensor->producer()->body()).initializer().Init(ret);  
      }
      Stream stream(Device(kCPU), NDArray::DEFAULT_STREAM);
      stream.Sync();
      return ret;
    } else {
      // The op has been instantiated in the current active graph. Let the executable graph handle it.
      if (!it_1->second->producer()->device_group().contains(impl::comm::GetLocalDevice())) {
        HT_LOG_TRACE << "The data is not locate at local executable graph, return an empty NDArray.";
        return NDArray::empty(tensor->shape(), Device(kCPU), tensor->dtype(), kBlockingStream);
      }
      auto ret = Graph::GetDetachedVariableData(it_1->second);
      Stream stream(Device(kCPU), NDArray::DEFAULT_STREAM);
      stream.Sync();
      return ret;
    }  
  } else {
    auto ret = NDArray::empty(tensor->shape(), Device(kCPU), tensor->dtype(), kBlockingStream);
    auto it = _add_on_inits.find(tensor->id());
    // Note _add_on_inits has a higher priority than the original tensor initializer.
    if (it != _add_on_inits.end()) {
      HT_LOG_TRACE << "No active exec graph yet. The data is reset, " 
        << "so we get the data of the variable from the DefineAndRun graph.";
      it->second->Init(ret);
    } else {
      HT_LOG_TRACE << "No active exec graph yet, "
        << "so we get the data of the variable from its initializer.";
      if (tensor->has_distributed_states())
        dynamic_cast<ParallelVariableOpImpl&>(tensor->producer()->body()).initializer().Init(ret);
      else
        dynamic_cast<VariableOpImpl&>(tensor->producer()->body()).initializer().Init(ret);  
    }
    Stream stream(Device(kCPU), NDArray::DEFAULT_STREAM);
    stream.Sync();
    return ret;
  }
}

DeviceGroup DefineAndRunGraph::GetVariableDeviceGroupInner(const Tensor& tensor) {
  auto& device_group = tensor->producer()->device_group();
  HT_RUNTIME_ERROR_IF(device_group.empty()) << "You are getting an empty device group, please ensure you have set "
    << tensor->producer() << " a device group before!";
  return device_group;
}

// 推导define graph在cur_strategy_id下的pipeline构造
void DefineAndRunGraph::DeducePipeline(size_t cur_strategy_id) {
  auto old_strategy_id = CUR_STRATEGY_ID;
  CUR_STRATEGY_ID = cur_strategy_id;
  std::unordered_map<Device, int32_t> device_to_p2pline_idx_map;
  std::unordered_map<int32_t, std::vector<Device>> p2plines;
  std::unordered_map<int32_t, DeviceGroupList> pipelines;
  int32_t total_p2pline = -1;
  int32_t total_pipeline = std::numeric_limits<int32_t>::max();
  // 得到有多少条p2pline以及每条p2pline所含的gpu
  // 注意这一步里的p2pline并不是最终的pipeline
  // 其还需要经过merge操作
  // 最终merge后的pipeline个数取决于最小的dup数量
  // 以dp2tp2pp2为例
  // bias的dup为4但weight的dup为2
  // 那么在一开始会生成四条pipeline但实际最终merge后pipeline条数为2
  // 考虑到更复杂的混合并行也应该是如此
  // 其本质是除了dup外就是split而split一定不能使用不同的feed dict的data
  // 因此做split的参数最终一定要求是在同一个pipeline中
  // 再扫一遍得到每条pipeline
  for (const auto& op : _ops_with_device_groups) {
    // deduce pipeline的stages时只需要用到模型的parameters
    if (_parameter_ops.find(op->id()) == _parameter_ops.end()) {
      continue;
    }
    auto& device_group = op->device_group();
    auto& ds = op->output(0)->cur_distributed_states();
    // HT_LOG_INFO << op << " device group: " << device_group << " and ds: " << ds.ds_info();
    HT_ASSERT(!device_group.empty() && !ds.is_none())
      << "device group & ds of " << op << " shouldn't be empty";
    auto num_devices = device_group.num_devices();
    total_pipeline = std::min(total_pipeline, ds.states(-1));
    if (total_p2pline == -1) {
      total_p2pline = num_devices;
    } else {
      HT_ASSERT(total_p2pline == num_devices)
        << "currently assume all devices will participate in the pipeline parallel"
        << ", and each parameter locates at a same amount of devices";
    }
    if (ds.states(-1) != 1) {
      HT_LOG_WARN_IF(ds.order(0) != -1) 
        << op << " is a parameter, which is suggested to put dup at the first position in the ds order sequence"
        << ", but now the ds is: " << ds.ds_info();
    }
    for (int32_t i = 0; i < num_devices; i++) {
      auto state_index = ds.map_device_to_state_index(i);
      // 按照split倒序然后dup的order找到其所在的p2pline
      std::vector<int32_t> keys;
      for (const auto& kv : state_index) {
        keys.emplace_back(kv.first);
      }
      std::sort(keys.rbegin(), keys.rend());
      int32_t interval = 1;
      int32_t new_idx = 0;
      for (int32_t key : keys) {
        new_idx += state_index[key] * interval;
        interval *= ds.states(key);
      }
      if (device_to_p2pline_idx_map.find(device_group.get(i)) != device_to_p2pline_idx_map.end()) {
        HT_ASSERT(device_to_p2pline_idx_map[device_group.get(i)] == new_idx)
          << "device " << device_group.get(i) << " is in two p2plines, which will cause chaos"
          << ", the existing p2pline is " << device_to_p2pline_idx_map[device_group.get(i)]
          << " and the new p2pline is " << new_idx; 
      } else {
        device_to_p2pline_idx_map[device_group.get(i)] = new_idx;
      }
      if (p2plines.find(new_idx) == p2plines.end()) {
        p2plines[new_idx] = std::vector<Device>{device_group.get(i)};
      } else {
        bool repetitive_device = false;
        for (const auto& device : p2plines[new_idx]) {
          if (device == device_group.get(i)) {
            repetitive_device = true;
            break;
          }
        }
        if (!repetitive_device) {
          p2plines[new_idx].emplace_back(device_group.get(i));
        }
      }
    }
  }
  // merge相邻的p2pline
  // 形成最终的pipeline
  HT_ASSERT(total_p2pline % total_pipeline == 0)
    << "total number of p2pline should be divided by total number of pipeline";
  int32_t merge_ratio = total_p2pline / total_pipeline;
  for (int32_t i = 0; i < total_pipeline; i++) {
    pipelines[i] = DeviceGroupList();
    int32_t total_stage = -1;
    std::vector<std::vector<Device>> stages;
    for (int32_t j = i * merge_ratio; j < (i + 1) * merge_ratio; j++) {
      if (total_stage == -1) {
        total_stage = p2plines[j].size();
        for (const auto& device : p2plines[j]) {
          stages.emplace_back(std::vector<Device>{device});
        }
      } else {
        HT_ASSERT(total_stage == p2plines[j].size())
          << "can't merge p2plines with different stages";
        for (int32_t stage_num = 0; stage_num < total_stage; stage_num++) {
          stages.at(stage_num).emplace_back(p2plines[j][stage_num]);
        }
      }
    }
    for (const auto& stage : stages) {
      pipelines[i].emplace_back(DeviceGroup(stage));
    }
  }
  // 记录当前strategy下的device到pipeline映射
  for (const auto& kv : device_to_p2pline_idx_map) {
    const auto& device = kv.first;
    const auto& p2pline_idx = kv.second;
    const auto pipeline_idx = p2pline_idx / merge_ratio;
    _multi_pipeline_maps[CUR_STRATEGY_ID][device] = pipelines[pipeline_idx];
  }
  CUR_STRATEGY_ID = old_strategy_id;
}

// 推导define graph的shape plan
// 以及exec graph的exec shape plan
// 请注意二者的区别
// 前者是用来进行plan匹配的（虽然目前feed dict固定实际上只用记录feed dict的shape）
// 后者是实际exec graph执行时runtime ctx要用到的
void DefineAndRunGraph::DeduceShapePlan(ExecGraphPlan& exec_graph_plan,
                                        const FeedDict& feed_dict,
                                        Tensor2ShapeMap& feed_dict_shape) {
  // *the logic of inferring the very first shape plan is in Instantiate()
  // that is because MakeOp can handle most of the cases automatically
  // InferShapePlan just aims to expand the shape plan pool for the data packing setting
  auto local_device = hetu::impl::comm::GetLocalDevice(); // debug use
  Tensor2ShapeMap shape_plan;
  Tensor2ShapeMap exec_shape_plan;
  RuntimeContext runtime_ctx{};
  // 扫描global topo并推导新的shape plan
  for (auto& op_ref : exec_graph_plan.global_topo) {
    auto& op = op_ref.get();
    // 设置placeholder（也有可能是中间的算子——具体要看feed_dict喂的是什么算子）的symbolic shape
    bool handle_feed_dict_op = Operator::all_output_tensors_of(op, [&](Tensor& tensor) {
      auto it = feed_dict.find(tensor->id());
      if (it != feed_dict.end()) {
        if (tensor->symbolic() && is_SyShape_leaf(tensor->symbolic_shape())) {
          tensor->set_symbolic_shape(feed_dict_shape[tensor->id()]);
          HT_LOG_DEBUG << local_device << ": set symbolic shape of " << op 
            << " feed_dict tensor to " << feed_dict_shape[tensor->id()];
        }
        shape_plan[tensor->id()] = feed_dict_shape[tensor->id()];
        return true;
      }
      return false;
    });
    if (handle_feed_dict_op) {
      continue;
    }
    HTShapeList input_shapes;
    input_shapes.reserve(op->num_inputs());
    for (const auto& input : op->inputs()) {
      auto it = shape_plan.find(input->id());
      HT_ASSERT(it != shape_plan.end()) 
        << "Something wrong, can't find the input shape from the current shape plan!";
      input_shapes.push_back(it->second);
    }
    HTShapeList output_shapes = op->InferShape(input_shapes, runtime_ctx);
    auto output_shapes_size = output_shapes.size();
    for (size_t i = 0; i < output_shapes_size; i++) {
      // 设置symbolic shape叶子节点的shape
      // 其相关联的非叶子的symbolic shape可以直接由计算链条获得新的shape
      if (op->output(i)->symbolic()) {
        HT_LOG_TRACE << local_device << ": op " << op 
          << " output " << i << " has " << op->output(i)->symbolic_shape();
        if (is_SyShape_leaf(op->output(i)->symbolic_shape())) {
          op->output(i)->set_symbolic_shape(output_shapes[i]);
          HT_LOG_TRACE << local_device << ": set symbolic shape of " << op 
            << " output " << i << " to " << output_shapes[i];
        }
      }
      HT_LOG_TRACE << local_device << ": " << op->output(i) << " shape " << output_shapes[i];
      auto it = shape_plan.find(op->output(i)->id());
      HT_ASSERT(it == shape_plan.end()) 
        << "Something wrong, the output shape should't exist in the current shape plan";
      shape_plan.insert(std::make_pair(op->output(i)->id(), std::move(output_shapes[i]))); // move constructor
    }
  }
  // define graph中已经推导得到的shape
  for (const auto& kv : exec_graph_plan.tensor_to_exec_tensor_mapping) {
    if (kv.second->producer()->num_outputs() == 0) {
      // 说明该tensor只是extra linker而并不会具有shape
      // 比如GroupOp
      continue;
    }
    auto it = shape_plan.find(kv.first);
    HT_ASSERT(it != shape_plan.end())
      << "can't find shape of tensor " << kv.second << " in the shape plan";
    exec_shape_plan[kv.second->id()] = it->second;
  }
  // exec graph中还有一些新增的shape
  for (const auto& exec_tensor : exec_graph_plan.exec_graph->_record_exec_tensors) {
    auto& exec_op = exec_tensor->producer();
    HTShapeList exec_input_shapes;
    exec_input_shapes.reserve(exec_op->num_inputs());
    for (const auto& exec_input : exec_op->inputs()) {
      auto it = exec_shape_plan.find(exec_input->id());
      HT_ASSERT(it != exec_shape_plan.end()) 
        << "Something wrong, can't find the input shape of " << exec_input
        << " from the current exec shape plan!";
      exec_input_shapes.push_back(it->second);
    }
    HTShapeList exec_output_shapes = exec_op->InferShape(exec_input_shapes, runtime_ctx);
    auto exec_output_shapes_size = exec_output_shapes.size();
    for (size_t i = 0; i < exec_output_shapes_size; i++) {
      if (exec_op->output(i)->symbolic()) {
        if (is_SyShape_leaf(exec_op->output(i)->symbolic_shape())) {
          exec_op->output(i)->set_symbolic_shape(exec_output_shapes[i]);
        }
      }
      exec_shape_plan.insert(std::make_pair(exec_op->output(i)->id(), std::move(exec_output_shapes[i]))); // move constructor
    }
  }
  exec_graph_plan.shape_plan_pool.emplace_back(std::move(shape_plan));
  exec_graph_plan.exec_graph->AddShapePlan(std::move(exec_shape_plan));
}

void DefineAndRunGraph::Instantiate(OpRefList&& global_topo,
                                    Tensor2ShapeMap&& shape_plan) {

  // deprecated: Test Case - 手动切换并行方案（验证切换时间）
  char* env = std::getenv("HETU_PARALLEL_CHANGE_TEST");
  if (env != nullptr) {
    if (std::string(env) == "COST" && change_parallel_test_case >= 1) {
      InstantiateTestCase(global_topo, shape_plan);
      change_parallel_test_case += 1;
      return;
    }
  }

  // initializations of the exec plan
  auto exec_graph_num = _exec_graph_plan_pool.size();
  Tensor2ShapeMap exec_shape_plan;
  Op2OpMap op_to_exec_op_mapping;
  Tensor2TensorMap tensor_to_exec_tensor_mapping;
  auto origin_param_buffer = std::make_shared<ParamBuffer>("origin_param_buffer");
  auto transfer_param_buffer = std::make_shared<ParamBuffer>("transfer_param_buffer");
  auto current_grad_buffer = std::make_shared<ParamBuffer>("current_grad_buffer");
  auto grad_concat_buffer = std::make_shared<ParamBuffer>("grad_concat_buffer");
  auto accumulate_grad_buffer = std::make_shared<ParamBuffer>("accumulate_grad_buffer");
  Tensor2TensorMap transfer_map;
  Tensor2TensorMap grad_map;
    
  exec_shape_plan.reserve(shape_plan.size());
  op_to_exec_op_mapping.reserve(_init_capacity);
  tensor_to_exec_tensor_mapping.reserve(_init_capacity);

  // initializations of the exec graph
  auto local_device = hetu::impl::comm::GetLocalDevice();
  auto exec_graph = Graph::_make_new_graph<ExecutableGraph>(name() + "_executable_" + std::to_string(exec_graph_num));
  exec_graph->NUM_STRATEGY = NUM_STRATEGY;
  exec_graph->CUR_STRATEGY_ID = CUR_STRATEGY_ID;
  // HT_LOG_INFO << local_device << ": instantiate " << exec_graph->name();
  Graph::push_graph_ctx(exec_graph->id());

  // assign pp stages
  if (_multi_pipeline_maps.find(CUR_STRATEGY_ID) == _multi_pipeline_maps.end()) {
    _multi_pipeline_maps[CUR_STRATEGY_ID] = Device2PipelineMap();
    DeducePipeline(CUR_STRATEGY_ID);
  }
  exec_graph->SetPipeline(_multi_pipeline_maps[CUR_STRATEGY_ID]);

  auto get_exec_input = [&](const Tensor& input) -> Tensor {
    auto it = tensor_to_exec_tensor_mapping.find(input->id());
    HT_RUNTIME_ERROR_IF(it == tensor_to_exec_tensor_mapping.end())
      << "Cannot find the executable version of Tensor " << input;
    return it->second;
  };

  // todo: just use multi_ds[cur_strategy_id] which was deduced in define_and_run_graph
  // executable_graph needn't deduce states again!
  auto handle_exec_output = [&](Tensor& tensor, Tensor& exec_tensor) -> void {
    HT_LOG_TRACE << "handle mapping of tensor " << tensor->id() << " " << tensor;
    // 1)、assign tensor mapping
    tensor_to_exec_tensor_mapping[tensor->id()] = exec_tensor;
    // 2)、assign shape
    auto plan_it = shape_plan.find(tensor->id());
    // The shape plan will be expanded step by step
    if (plan_it != shape_plan.end()) {
      // *only feed dict will set_shape
      exec_tensor->set_shape(plan_it->second);
    } else {
      // other shapes will be fixed and just recorded
      shape_plan[tensor->id()] = exec_tensor->shape();
    }
    exec_shape_plan[exec_tensor->id()] = exec_tensor->shape();
    HT_LOG_TRACE << "assign exec tensor " << exec_tensor << " shape " << exec_tensor->shape();
    exec_tensor->set_is_grad(tensor->is_grad());
    // 3)、assign symbolic shape
    // here symbolic shape will only used in some op
    // such as slice & reshape
    // the tensor shape is fixed and recorded in the shape_plan
    // note that no tensor will have an unknown shape in the exec graph
    if (tensor->symbolic()) {
      exec_tensor->copy_symbolic_shape(tensor->symbolic_shape());
      if (is_SyShape_leaf(exec_tensor->symbolic_shape())) {
        exec_tensor->set_symbolic_shape(exec_tensor->shape());
      }
    }
    // 4)、assign distributed_states
    // just copy distributed_states here
    exec_tensor->set_multi_distributed_states(tensor->multi_distributed_states());
    // 5)、assign add on inits
    auto it = _add_on_inits.find(tensor->id());
    if (_run_level != RunLevel::TOPO && it != _add_on_inits.end()) {
      Graph::ResetVariableData(exec_tensor, *it->second);
      // 考虑要切换plan，仅第一次使用_add_on_inits
      // 之后会使用热切换
      _add_on_inits.erase(tensor->id());
    }
    // 6)、assign param 
    // 目前只是记录而并不会alloc
    if (_parameter_ops.find(tensor->producer()->id()) != _parameter_ops.end()
        && exec_tensor->producer()->device_group().contains(local_device)
        && tensor->requires_grad()) {
      origin_param_buffer->AddTensor(exec_tensor);
      /*
      exec_tensor->set_placement_group(exec_tensor->producer()->device_group());
      exec_tensor->set_placement(local_device);
      Graph::AllocVariableData(exec_tensor);
      */
    }
  };

  HT_LOG_DEBUG << "Instantiating a " << type() << " graph with global topo " << global_topo;
  for (auto& op_ref : global_topo) {
    auto& op = op_ref.get();
    HT_LOG_TRACE << "Creating an executable version of op " << op << " begin...";

    // 前处理
    // 1、获取exec op的inputs
    // 2、进行autocast
    TensorList exec_inputs, exec_in_deps;
    std::tie(exec_inputs, exec_in_deps) = Operator::transform_each_input_tensor(op, get_exec_input);

    // symbolic shape debug use
    /*
    HTShapeList exec_input_shapes;
    for (auto& exec_input : exec_inputs) {
      exec_input_shapes.push_back(exec_input->shape());
    }
    HT_LOG_INFO << "Exec op " << op << " with inputs " << exec_inputs << " and shapes " << exec_input_shapes;
    */

    auto autocast_id = AutoCast::cur_autocast_ctx();
    if (autocast_id != UINT64_MAX) {
      auto autocast = AutoCast::GetAutoCast(autocast_id);
      if (autocast.enabled()) {
        DataType datatype = DataType::UNDETERMINED;
        if (autocast.cast_type() != DataType::UNDETERMINED)
          datatype = autocast.cast_type();
        if (datatype != DataType::UNDETERMINED) {
          auto optype = op->type();
          if (is_optimizer_update_op(op) || is_host_to_device_op(op) || is_device_to_host_op(op) || is_data_transfer_op(op)) {
            // seems nothing to do
          } else {
            for (int i = 0; i < exec_inputs.size(); ++i) {
              if ((is_variable_op(exec_inputs[i]->producer()) || is_placeholder_op(exec_inputs[i]->producer())) &&
                  exec_inputs[i]->dtype() != datatype && 
                  (exec_inputs[i]->dtype() == DataType::BFLOAT16 ||
                  exec_inputs[i]->dtype() == DataType::FLOAT16 ||
                  exec_inputs[i]->dtype() == DataType::FLOAT32 ||
                  exec_inputs[i]->dtype() == DataType::FLOAT64)) {
                if (transfer_map.find(exec_inputs[i]->id()) != transfer_map.end()) {
                  HT_LOG_TRACE << "Map " << &transfer_map << " reuse: " << exec_inputs[i]->id() << " -> " << transfer_map[exec_inputs[i]->id()]->id();
                  exec_inputs[i] = transfer_map[exec_inputs[i]->id()];
                } else {
                  auto& exec_op = Graph::MakeOp(std::make_shared<DataTransferOpImpl>(datatype, exec_inputs[i]->device()),
                                  {exec_inputs[i]}, OpMeta().set(exec_inputs[i]->producer()->op_meta()).set_name(exec_inputs[i]->producer()->name() + "_autocast").set_is_deduce_states(false), *exec_graph);
                  HT_LOG_TRACE << "Map " << &transfer_map << " insert: " << exec_inputs[i]->id() << " -> " << exec_op->output(0)->id();
                  // we have to set the exec shape plan manually before the initialization of the plan
                  exec_shape_plan[exec_op->output(0)->id()] = exec_op->output(0)->shape();
                  exec_graph->_record_exec_tensors.emplace_back(exec_op->output(0));
                  exec_op->output(0)->set_multi_distributed_states(op->input(i)->multi_distributed_states()); // walkaround: set here by hand
                  if (_parameter_ops.find(op->input(i)->producer()->id()) != _parameter_ops.end()
                      && exec_inputs[i]->producer()->device_group().contains(local_device)) {
                    transfer_param_buffer->AddTensor(exec_op->output(0));
                  }
                  transfer_map[exec_inputs[i]->id()] = exec_op->output(0);
                  exec_inputs[i] = exec_op->output(0);
                }
              }
            }
          }
        }
      }
    }

    // 核心部分
    // only deduce multi ds for define_and_run_graph, and copy directly for executable_graph
    auto& exec_op = Graph::MakeOp(
      op->_body, std::move(exec_inputs),
      OpMeta().set(op->op_meta()).set_is_deduce_states(false).set_extra_deps(std::move(exec_in_deps)),
      *exec_graph);

    // 后处理
    // 1、建立op和exec_op的映射
    // 2、设置tensor的shape和distributed_states
    // 3、标记parameter并给即将创建的exec graph预先设置ParamBuffer
    // 4、给grad设置placement和buffer
    op_to_exec_op_mapping[op->id()] = exec_op;
    Operator::for_each_output_tensor_pair(op, exec_op, handle_exec_output);
    if (_parameter_ops.find(op->id()) != _parameter_ops.end()) {
      Graph::MarkAsParameter(exec_op);
    }
    if (is_optimizer_update_op(exec_op)) {
      Tensor& param = op->input(0);
      Tensor& exec_param = exec_op->input(0);
      Tensor& exec_grad = exec_op->input(1);
      HT_ASSERT(exec_graph->_parameter_ops.find(exec_param->producer()->id()) != exec_graph->_parameter_ops.end())
        << "optimizer op " << exec_op << " input 0 " << exec_param << " is not a parameter";
      // zero属性已经类似multi_ds一样设置成了list
      /*
      auto zero = (param->get_distributed_states().get_dim(-1) > 1) && param->get_distributed_states().zero();
      auto adam_op_interface = std::dynamic_pointer_cast<AdamOpImpl>(exec_op->_body);
      if (adam_op_interface) {
        adam_op_interface->set_zero(zero);
      }
      */
      // 热切换接口需要提前设置一些grad的信息
      exec_grad->producer()->set_device_groups(exec_param->producer()->device_groups());
      if (exec_grad->producer()->device_group().contains(local_device)) {
        current_grad_buffer->AddTensor(exec_grad);
        accumulate_grad_buffer->AddTensor(exec_grad);
        exec_grad->set_placement_group(exec_param->producer()->device_group());
        exec_grad->set_placement(local_device);
        HT_LOG_TRACE << "local grad " << exec_grad << " ds states = " << exec_grad->get_distributed_states().get_states() 
          << " and order = " << exec_grad->get_distributed_states().get_order();
      }
      grad_map[exec_param->id()] = exec_grad;
    }
    HT_LOG_TRACE << "Creating an executable version of op " << op << " end...";
  }

  // 遍历current_grad_buffer的所有tensor，把最大的tensor加入到grad_concat_buffer中
  auto& grad_tensors = current_grad_buffer->tensor_list();
  if (!grad_tensors.empty()) {
    auto max_grad_it = std::max_element(grad_tensors.begin(), grad_tensors.end(), 
      [](const Tensor& a, const Tensor& b) {
        return a->numel() < b->numel();
      });
    grad_concat_buffer->AddTensor(*max_grad_it);
  }

  // assign fw_op_id map
  for (auto& op_ref : global_topo) {
    auto& op = op_ref.get();
    auto& exec_op = op_to_exec_op_mapping[op->id()];
    if (op->fw_op_id() != -1) {
      exec_op->set_fw_op_id(op_to_exec_op_mapping[op->fw_op_id()]->id());
    } 
  }
  
  // assign initial shape plan
  exec_graph->AddShapePlan(std::move(exec_shape_plan));

  // assign param buffer, grad buffer and transfer map
  exec_graph->_origin_param_buffer = std::move(origin_param_buffer);
  exec_graph->_transfer_param_buffer = std::move(transfer_param_buffer);
  exec_graph->_current_grad_buffer = std::move(current_grad_buffer);
  exec_graph->_grad_concat_buffer = std::move(grad_concat_buffer);
  exec_graph->_accumulate_grad_buffer = std::move(accumulate_grad_buffer);
  exec_graph->_transfer_map = std::move(transfer_map);
  exec_graph->_grad_map = std::move(grad_map);
  
  // wrap up all of this as an exec graph plan
  _exec_graph_plan_pool.emplace_back(std::move(exec_graph), 
                                     std::move(op_to_exec_op_mapping),
                                     std::move(tensor_to_exec_tensor_mapping),
                                     std::move(global_topo),
                                     std::vector<Tensor2ShapeMap>{std::move(shape_plan)},
                                     CUR_STRATEGY_ID);

  Graph::pop_graph_ctx();

  // deprecated: Test Case - 手动切换并行方案
  if (env != nullptr) {
    if (std::string(env) == "PRECISION" || std::string(env) == "COST") {
      change_parallel_test_case += 1;
    }
  }
}

// 每次调用run都会从当前的define graph中
// 生成/使用之前生成过的一个exec graph
// 而只有当：
// 1、并行策略 2、fetch的tensor 
// 与cache的某一个重合时，才会复用
// 目前的写法下，我们认为并行策略已经在python端选择好了然后再传进来
// 2024.3.6 update:
// 目前一个exec graph支持多个shape plan
// 即允许feed_dict的shape（包括batch_size以及seq_len等）可变
NDArrayList DefineAndRunGraph::Run(const Tensor& loss, const TensorList& fetches,
                                   const FeedDict& feed_dict, const int num_micro_batches,
                                   const int cur_strategy_id, RunLevel run_level, const double grad_scale) {
  _run_level = run_level;
  CUR_STRATEGY_ID = static_cast<size_t>(cur_strategy_id);
  auto local_device = hetu::impl::comm::GetLocalDevice(); // only for debug use
  HT_LOG_DEBUG << local_device << ": [Graph Plan] obtain exec graph begin...";

  // get feed dict shape
  Tensor2ShapeMapList feed_dict_shape_list(num_micro_batches);
  for (const auto& kv : feed_dict) {
    if (kv.second.size() == 0) 
      continue; 
    if (kv.second.size() == 1) {
      // all micro batches have the same shape
      auto micro_batches = NDArray::split(kv.second[0], num_micro_batches);
      for (auto& feed_dict_shape : feed_dict_shape_list) {
        feed_dict_shape[kv.first] = micro_batches[0]->shape();
      }
    } else {
      HT_ASSERT(kv.second.size() == num_micro_batches);
      for (int i = 0; i < num_micro_batches; i++) {
        feed_dict_shape_list[i][kv.first] = kv.second[i]->shape();
      }
    }
  }

  size_t next_active_exec_plan;
  std::vector<size_t> next_active_shape_plan_list(num_micro_batches);
  int64_t micro_batch_idx = 0;
  size_t exec_plan_pool_size = _exec_graph_plan_pool.size();
  bool in_exec_plan_pool = false;
  for (size_t i = 0; i < exec_plan_pool_size; i++)  {
    const auto& exec_graph_plan = _exec_graph_plan_pool[i];
    bool exec_plan_matched = true;
    // 先看strategy匹配不
    if (static_cast<size_t>(cur_strategy_id) != exec_graph_plan.strategy_id) {
      exec_plan_matched = false;
    }
    // 再看fetch匹配不
    for (const auto& fetch : fetches) {
      if (std::find(exec_graph_plan.fetches.begin(), exec_graph_plan.fetches.end(), fetch) == exec_graph_plan.fetches.end()) {
        HT_LOG_TRACE << local_device << ": exec_graph_plan fetches are " << exec_graph_plan.fetches 
          << " and the mismatch fetch is " << fetch;
        exec_plan_matched = false;
        break;
      }
    }
    if (exec_plan_matched) {
      HT_LOG_TRACE << local_device << ": plan matched";
      in_exec_plan_pool = true;
      next_active_exec_plan = i;
      break;
    }
  }

  // 需要创建一个新的exec graph
  // 用当前feed dict的shape先初始化一套shape plan
  // 作为该exec graph的shape plan pool里的第一个
  if (!in_exec_plan_pool) {
    HT_LOG_DEBUG << local_device << ": [Graph Plan] add a new exec graph to the pool begin...";
    Tensor2ShapeMap shape_plan;
    // 后续会由feed_dict的shape在MakeOp时推导出所有的shape
    for (const auto& kv : feed_dict) {
      shape_plan[kv.first] = feed_dict_shape_list[micro_batch_idx][kv.first];
    }
    auto is_feed_dict_op = [&](const Operator& op) -> bool {
      return Operator::all_output_tensors_of(op, [&](const Tensor& tensor) {
        return feed_dict.find(tensor->id()) != feed_dict.end();
      });
    };
    OpRefList global_topo = Graph::TopoSort(fetches, -1, is_feed_dict_op);
    HT_LOG_DEBUG << local_device << ": global topo of define graph is " << global_topo;
    // Instantiate会将新的exec_graph_plan加入pool中
    Instantiate(std::move(global_topo), std::move(shape_plan));
    // 补上fetches（其在instantiate中不需要用到，但是plan需要进行记录）
    auto& new_plan = _exec_graph_plan_pool.back();
    new_plan.fetches = fetches;
    // 新的exec plan就是exec plan pool中的最后一个
    next_active_exec_plan = _exec_graph_plan_pool.size() - 1;
    // 新的shape plan就是shape plan pool中的第一个
    next_active_shape_plan_list[micro_batch_idx] = 0;
    micro_batch_idx++; 
    HT_LOG_DEBUG << local_device << ": [Graph Plan] add a new shape plan and an exec graph to the pool end...";
  } 
  // 命中pool中已有的exec graph
  // 但可能feed dict不一样
  // 这种情况下我们不需要生成新的exec graph
  // 但需要推导新的shape plan
  for (auto idx = micro_batch_idx; idx < num_micro_batches; idx++) {
    auto& exec_graph_plan = _exec_graph_plan_pool[next_active_exec_plan];
    auto shape_plan_pool_size = exec_graph_plan.shape_plan_pool.size();
    bool in_shape_plan_pool = false;
    for (size_t i = 0; i < shape_plan_pool_size; i++) {
      const auto& shape_plan = exec_graph_plan.shape_plan_pool[i];
      bool shape_plan_matched = true;
      for (const auto& kv : feed_dict) {
        if (kv.second.size() == 0) continue;
        auto it = shape_plan.find(kv.first);
        // 1、有可能是feed_dict发生了改变（在依据global topo生成的shape plan中没有feed dict）
        // 2、有可能是feed_dict的shape发生了改变（shape对不上）
        HT_LOG_TRACE << local_device << ": shape plan is " << shape_plan << " and key to match is "
          << kv.first << ":" << feed_dict_shape_list[idx][kv.first];
        if (it == shape_plan.end() || it->second != feed_dict_shape_list[idx][kv.first]) {
          shape_plan_matched = false;
          break;
        }
      }
      if (shape_plan_matched) {
        in_shape_plan_pool = true;
        next_active_shape_plan_list[idx] = i;
        break;
      }
    }
    // 如果不在shape_plan_pool中
    // 需要推导新的shape plan
    if (!in_shape_plan_pool) {
      DeduceShapePlan(exec_graph_plan, feed_dict, feed_dict_shape_list[idx]);                                            
      // 新的shape plan就是shape plan pool中的最后一个
      next_active_shape_plan_list[idx] = exec_graph_plan.shape_plan_pool.size() - 1;
    }
  }

  // 需要切换exec graph
  if (!_is_active || _active_exec_plan != next_active_exec_plan) {
    HT_LOG_DEBUG << local_device << ": [Graph Plan] Context switch to the new exec plan begin...";
    // 热切换
    if (_is_active) {
      auto key = std::make_pair(_active_exec_plan, next_active_exec_plan);
      if (_param_switcher_pool.find(key) == _param_switcher_pool.end()) {
        _param_switcher_pool[key] = std::make_shared<SwitchExecGraph>(this, _active_exec_plan, next_active_exec_plan);
        _grad_switcher_pool[key] = std::make_shared<SwitchExecGraph>(this, _active_exec_plan, next_active_exec_plan);
      }
      // 旧的exec graph
      auto& old_exec_graph = _exec_graph_plan_pool[_active_exec_plan].exec_graph;
      // 默认的切换状态设置
      auto param_switch_mode = SWITCH_MODE::SWITCH_TRANSFER_PARAM;
      auto grad_switch_mode = SWITCH_MODE::SWITCH_ACCUMULATE_GRAD;
      auto param_switch_level = SWITCH_LEVEL::EXEC;
      auto grad_switch_level = SWITCH_LEVEL::EXEC;
      // 1、----- level设置 -----
      // 1)、topo前只能跟topo（算exec graph和switcher的topo）
      // 2)、alloc前只能跟topo或alloc或update（新的一轮开始）
      // 3)、grad后只能跟grad或update（grad要么不断累积要么更新掉）
      // 4)、update前都能跟
      // 其实原则就是有transfer param就切，没有就切origin param
      // 有accumulate grad就切，没有就不切
      if (_run_level == RunLevel::TOPO) {
        HT_ASSERT(old_exec_graph->_run_level == RunLevel::TOPO) 
          << "graph with RunLevel::TOPO should only follow behind graph with RunLevel::TOPO right now";
      }
      if (_run_level == RunLevel::ALLOC) {
        HT_ASSERT(old_exec_graph->_run_level == RunLevel::TOPO
                  || old_exec_graph->_run_level == RunLevel::ALLOC
                  || old_exec_graph->_run_level == RunLevel::UPDATE) 
          << "graph with RunLevel::ALLOC should only follow behind graph with RunLevel::TOPO or RunLevel::ALLOC or RunLevel::UPDATE right now";
      }
      if (old_exec_graph->_run_level == RunLevel::GRAD) {
        HT_ASSERT(_run_level == RunLevel::GRAD || _run_level == RunLevel::UPDATE) 
          << "graph with RunLevel::GRAD should only followed by graph with RunLevel::GRAD or RunLevel::UPDATE right now";
      }
      // 如果旧的exec graph只是建立topo
      // 其并没有产生param和grad
      if (old_exec_graph->_run_level == RunLevel::TOPO) {
        param_switch_level = SWITCH_LEVEL::TOPO;
        grad_switch_level = SWITCH_LEVEL::TOPO;
      }
      // 如果旧的exec graph只是alloc
      // 其并没有产生grad
      if (old_exec_graph->_run_level == RunLevel::ALLOC) {
        grad_switch_level = SWITCH_LEVEL::TOPO;
      }
      // 如果旧的exec graph是update
      // grad已经被消耗掉了
      if (old_exec_graph->_run_level == RunLevel::UPDATE) {
        grad_switch_level = SWITCH_LEVEL::TOPO;
      }
      // 2、----- mode设置 -----
      // 如果旧的exec graph没开AMP
      // 或者是刚刚进行了update（使得transfer param是空的）
      // 那么只能切换origin param buffer
      if (old_exec_graph->_transfer_param_buffer->IsEmpty()
          || (old_exec_graph->_run_level == RunLevel::UPDATE
              && param_switch_level == SWITCH_LEVEL::EXEC)) {
        param_switch_mode = SWITCH_MODE::SWITCH_ORIGIN_PARAM;
      }
      // 3、----- buffer释放 -----
      // 如果旧的exec graph是grad
      // 那么热切换需要释放之前的current grad buffer
      // 如果旧的exec graph是update
      // 那么热切换需要释放之前的transfer param buffer和current grad buffer
      if (old_exec_graph->_run_level == RunLevel::GRAD) {
        if (old_exec_graph->_use_current_grad_buffer) {
          if (!old_exec_graph->_current_grad_buffer->IsEmpty()) {
            HT_ASSERT(old_exec_graph->_current_grad_buffer->IsAllocated())
              << "old exec graph with RunLevel::GRAD should have allocated the current grad buffer";
            old_exec_graph->_current_grad_buffer->Free();
          }
        }
      }
      if (old_exec_graph->_run_level == RunLevel::UPDATE) {
        if (!old_exec_graph->_transfer_param_buffer->IsEmpty()) {
          HT_ASSERT(old_exec_graph->_transfer_param_buffer->IsAllocated())
            << "old exec graph with RunLevel::UPDATE should have allocated the transfer param buffer";
          old_exec_graph->_transfer_param_buffer->Free();
        }
        if (old_exec_graph->_use_current_grad_buffer) {
          if (!old_exec_graph->_current_grad_buffer->IsEmpty()) {
            HT_ASSERT(old_exec_graph->_current_grad_buffer->IsAllocated())
              << "old exec graph with RunLevel::UPDATE should have allocated the current grad buffer";
            old_exec_graph->_current_grad_buffer->Free();
          }
        }
      }
      /*
      for (auto& tensor : _exec_graph_plan_pool[next_active_exec_plan].exec_graph->_transfer_param_buffer->tensor_list()) {
        HT_LOG_INFO << local_device << ": transfer param " << tensor << " meta is " << tensor->meta() << " and device group is " << tensor->producer()->device_group()
          << " and ds is: " << tensor->get_distributed_states().ds_info();
      }
      for (auto& tensor : _exec_graph_plan_pool[next_active_exec_plan].exec_graph->_accumulate_grad_buffer->tensor_list()) {
        HT_LOG_INFO << local_device << ": accumulate grad " << tensor << " meta is " << tensor->meta() << " and device group is " << tensor->producer()->device_group()
          << " and ds is: " << tensor->get_distributed_states().ds_info();
      }
      */
      // 实际热切换
      // 目前已修改成async版本
      // 如果要改成非async的
      // 更改环境变量HETU_SWITCH_PROFILE低于TIME即可
      _param_switcher_pool[key]->SwitchParams(param_switch_mode, param_switch_level);
      _grad_switcher_pool[key]->SwitchParams(grad_switch_mode, grad_switch_level);
    }
    _is_active = true;
    _active_exec_plan = next_active_exec_plan;
    HT_LOG_DEBUG << local_device << ": [Graph Plan] Context switch to the new exec plan end...";
  }
  HT_LOG_DEBUG << local_device << ": [Graph Plan] obtain exec graph end...";

  // deprecated: Test Case - 手动切换并行方案（验证切换时间）
  char* env = std::getenv("HETU_PARALLEL_CHANGE_TEST");
  if (env != nullptr) {
    if (std::string(env) == "COST" && change_parallel_test_case >= 1) {
      return {};
    }
  }

  // 运行挑选出的active exec graph
  auto& exec_graph = _exec_graph_plan_pool[_active_exec_plan].exec_graph;
  auto& op_to_exec_op_mapping = _exec_graph_plan_pool[_active_exec_plan].op_to_exec_op_mapping;
  auto& tensor_to_exec_tensor_mapping = _exec_graph_plan_pool[_active_exec_plan].tensor_to_exec_tensor_mapping;
  auto& exec_loss = tensor_to_exec_tensor_mapping[loss->id()]; 
  TensorList exec_fetches;
  FeedDict exec_feed_dict;

  // 设置shape plan
  HT_LOG_DEBUG << exec_graph->name() << " use shape plan " << next_active_shape_plan_list;
  exec_graph->SetShapePlan(next_active_shape_plan_list[0]);
  exec_graph->SetShapePlanList(std::move(next_active_shape_plan_list));

  exec_fetches.reserve(fetches.size());
  for (const auto& fetch : fetches) {
    HT_ASSERT(tensor_to_exec_tensor_mapping.find(fetch->id()) != tensor_to_exec_tensor_mapping.end())
      << "can't find fetch tensor " << fetch << " in the mapping";
    exec_fetches.push_back(tensor_to_exec_tensor_mapping[fetch->id()]);
  }
  exec_feed_dict.reserve(feed_dict.size());
  for (const auto& kv : feed_dict) {
    if (tensor_to_exec_tensor_mapping.find(kv.first) == tensor_to_exec_tensor_mapping.end()) {
      HT_LOG_DEBUG << "feed tensor " << kv.first << " is not used in the exec graph"
        << ", so we just skipped it";
      continue;
    }
    exec_feed_dict[tensor_to_exec_tensor_mapping[kv.first]->id()] = kv.second;
  }
  // 验证mempool是否能释放干净
  // GetCUDAProfiler(local_device)->PrintCurrMemoryInfo(name() + " before empty cache");
  // hetu::impl::ProfileAfterEmptyAllCUDACache(local_device);
  HT_LOG_DEBUG << exec_graph->name() << " start running..." ;
  Graph::push_graph_ctx(exec_graph->id()); // 防止exec graph run内部MakeOp时忘记加
  auto ret = exec_graph->Run(exec_loss, exec_fetches, exec_feed_dict, num_micro_batches, 
                             cur_strategy_id, run_level, grad_scale);
  Graph::pop_graph_ctx();
  // 释放graph切换相关的event
  exec_graph->_switch_param_events.clear();
  exec_graph->_switch_grad_events.clear();
  // 验证mempool是否能释放干净
  // hetu::impl::ProfileAfterEmptyAllCUDACache(local_device);
  // GetCUDAProfiler(local_device)->PrintCurrMemoryInfo(name() + " after empty cache");
  return ret;
}

// TODO: merge two `Run` func
NDArrayList DefineAndRunGraph::Run(const TensorList& fetches,
                                   const FeedDict& feed_dict) {
  HT_RUNTIME_ERROR << "NotImplementedError";
  /*
  bool has_uninstantiated_ops =
    std::any_of(fetches.begin(), fetches.end(), [&](const Tensor& fetch) {
      return _op_to_exec_op_mapping.find(fetch->producer_id()) ==
        _op_to_exec_op_mapping.end();
    });
  if (has_uninstantiated_ops)
    Instantiate();
  TensorList exec_fetches;
  exec_fetches.reserve(fetches.size());
  for (const auto& fetch : fetches) {
    exec_fetches.push_back(_tensor_to_exec_tensor_mapping[fetch->id()]);
  }
  FeedDict exec_feed_dict;
  exec_feed_dict.reserve(feed_dict.size());
  for (const auto& kv : feed_dict)
    exec_feed_dict[_tensor_to_exec_tensor_mapping[kv.first]->id()] = kv.second;
  return _exec_graph->Run(exec_fetches, exec_feed_dict);
  */
}

} // namespace graph
} // namespace hetu
