#include "hetu/graph/headers.h"
#include "hetu/graph/define_by_run_graph.h"
#include "hetu/graph/define_and_run_graph.h"
#include "hetu/graph/eager_graph.h"
#include "hetu/graph/executable_graph.h"
#include "hetu/graph/ops/Contiguous.h"
#include "hetu/graph/ops/ones_like.h"
#include "hetu/graph/ops/sum.h"
#include "hetu/graph/ops/Contiguous.h"
#include "hetu/impl/communication/comm_group.h"
#include <thread>

namespace hetu {
namespace graph {

std::once_flag Graph::_init_flag;
std::vector<std::shared_ptr<Graph>> Graph::_global_graphs;
std::unordered_map<GraphName, std::shared_ptr<Graph>> Graph::_name_to_graphs;
std::shared_ptr<Graph> Graph::_default_eager_graph;
std::shared_ptr<Graph> Graph::_default_define_by_run_graph;
std::shared_ptr<Graph> Graph::_default_define_and_run_graph;
thread_local std::stack<GraphId> Graph::_cur_graph_ctx;

GraphId Graph::_next_graph_id() {
  static std::atomic<GraphId> _global_graph_id{0};
  return _global_graph_id++;
}

void Graph::Init() {
  // exit handler
  auto status = std::atexit([]() {
    HT_LOG_DEBUG << "Clearing and destructing all graphs...";
    Graph::_name_to_graphs.clear();
    Graph::_default_eager_graph = nullptr;
    Graph::_default_define_by_run_graph = nullptr;
    Graph::_default_define_and_run_graph = nullptr;
    for (auto& graph : Graph::_global_graphs) {
      if (graph == nullptr)
        continue;
      graph->Clear();
    }
    Graph::_global_graphs.clear();
    HT_LOG_DEBUG << "Destructed all graphs";
  });
  HT_ASSERT(status == 0)
      << "Failed to register the exit function for memory pools.";

  auto concurrency = std::thread::hardware_concurrency();
  Graph::_global_graphs.reserve(MIN(concurrency, 16) * 2);
  Graph::_name_to_graphs.reserve(MIN(concurrency, 16) * 2);
  Graph::_default_eager_graph =
    Graph::_make_new_graph<EagerGraph>("default_eager");
  Graph::_default_define_by_run_graph =
    Graph::_make_new_graph<DefineByRunGraph>("default_define_by_run");
  Graph::_default_define_and_run_graph =
    Graph::_make_new_graph<DefineAndRunGraph>("default_define_and_run");
}

Operator& Graph::MakeOp(std::shared_ptr<OpInterface> body, TensorList inputs,
                        OpMeta op_meta) {
  Graph::InitOnce();
  if (inputs.empty() && op_meta.extra_deps.empty()) {
    HT_VALUE_ERROR_IF(Graph::_cur_graph_ctx.empty())
      << "The target graph must be explicitly passed or enqueued to ctx "
      << "when making a new op with zero inputs";
    HT_LOG_TRACE << "Make variable op on a " << Graph::GetGraph(Graph::_cur_graph_ctx.top()).type() << " graph";
    return MakeOp(std::move(body), std::move(inputs), std::move(op_meta),
                  Graph::GetGraph(Graph::_cur_graph_ctx.top()));
  } else {
    GraphId target_graph_id = std::numeric_limits<GraphId>::max();
    bool input_graph_changed = false;
    auto find_target_graph = [&](const Tensor& input) mutable {
      auto& input_graph = Graph::GetGraph(input->graph_id());
      if (target_graph_id == std::numeric_limits<GraphId>::max()) {
        target_graph_id = input->graph_id();
      } else if (target_graph_id != input->graph_id()) {
        input_graph_changed = true;
        if (input_graph.type() == GraphType::DEFINE_BY_RUN) {
          target_graph_id = input->graph_id();
        }
      }
      HT_VALUE_ERROR_IF(input_graph_changed &&
                        input_graph.type() != GraphType::EAGER &&
                        input_graph.type() != GraphType::DEFINE_BY_RUN && 
                        input_graph.type() != GraphType::DEFINE_AND_RUN)
        << "The target graph must be explicitly passed "
        << "when making new op to a " << input_graph.type() << " graph";
    };
    for (auto& input : inputs)
      find_target_graph(input);
    for (auto& in_dep : op_meta.extra_deps)
      find_target_graph(in_dep);
    return MakeOp(std::move(body), std::move(inputs), std::move(op_meta),
                  Graph::GetGraph(target_graph_id));
  }
}

Operator& Graph::MakeOp(std::shared_ptr<OpInterface> body, TensorList inputs,
                        OpMeta op_meta, Graph& graph) {
  Graph::InitOnce();
  return graph.MakeOpInner(std::move(body), std::move(inputs),
                           std::move(op_meta));
}

TensorList Graph::Gradients(const TensorList& ys, const TensorList& xs,
                            const TensorList& grad_ys, int32_t num_ops_hint) {
  for (const auto& y : ys) {
    HT_VALUE_ERROR_IF(!y.is_defined()) << "Passed an undefined tensor";
    HT_VALUE_ERROR_IF(y->is_out_dep_linker())
      << "Cannot compute the gradient of " << y << " with operator "
      << y->producer()->type();
  }

  // Fill gradients
  TensorList filled_grads;
  filled_grads.reserve(ys.size());
  if (grad_ys.empty()) {
    for (const auto& y : ys) {
      // TODO: check whether requires grad
      filled_grads.emplace_back(MakeOnesLikeOp(y));
      filled_grads.back()->set_is_grad(true);
      filled_grads.back()->producer()->set_fw_op_id(y->producer()->id());
    }
  } else {
    HT_VALUE_ERROR_IF(ys.size() != grad_ys.size())
      << "Provided " << grad_ys.size() << " gradients for " << ys.size()
      << " tensors";
    for (size_t i = 0; i < ys.size(); i++) {
      if (!grad_ys[i].is_defined()) {
        filled_grads.emplace_back(MakeOnesLikeOp(ys[i]));
      } else {
        filled_grads.push_back(grad_ys[i]);
      }
      filled_grads.back()->set_is_grad(true);
      filled_grads.back()->producer()->set_fw_op_id(ys[i]->producer()->id());      
    }
  }

  Tensor2TensorListMap tensor_to_grads;
  Tensor2TensorMap tensor_to_reduced_grad;
  for (size_t i = 0; i < ys.size(); i++) {
    auto it = tensor_to_grads.find(ys[i]->id());
    if (it == tensor_to_grads.end())
      tensor_to_grads[ys[i]->id()] = {filled_grads[i]};
    else
      it->second.push_back(filled_grads[i]);
  }

  auto reduce_grad = [](const TensorList& unreduced_grads, bool is_fw_comm_op) -> Tensor {
    TensorList filtered;
    filtered.reserve(unreduced_grads.size());
    for (const auto& grad : unreduced_grads)
      if (grad.is_defined())
        filtered.push_back(grad);
    if (filtered.empty()) {
      return Tensor();
    } else if (filtered.size() == 1) {
      return filtered.front();
    } else {
      // Question: How to set op_meta properly?
      // if grad in filtered are all allreduce/reduce-scatter
      // should use last bw grad here, correspond to first use in fw, 
      // if the tensor was shared in different pp stages
      OpId fw_op_id = filtered.back()->producer()->fw_op_id();
      // comm op的multi ds中只要有一组满足is_all_allreduce || is_all_reduce_scatter的条件, 
      // 就要走先sum再comm的路径
      auto& graph = filtered[0]->graph();
      DistributedStatesList multi_dst_ds;
      bool is_need_sum_before_reduce = false;
      for (size_t cur_strategy_id = 0; cur_strategy_id < graph.NUM_STRATEGY; cur_strategy_id++) {
        graph.CUR_STRATEGY_ID = cur_strategy_id;
        bool is_all_allreduce = true;
        bool is_all_reduce_scatter = true;
        for (const auto& grad : filtered) {
          if (is_comm_op(grad->producer())) {
            auto& comm_op_impl = reinterpret_cast<CommOpImpl&>(grad->producer()->body());
            uint64_t comm_type = comm_op_impl.get_comm_type(grad->producer());
            if (comm_type != ALL_REDUCE_OP) {
              is_all_allreduce = false;
            }
            if (comm_type != REDUCE_SCATTER_OP) {
              is_all_reduce_scatter = false;
            }
            if (!is_all_allreduce && !is_all_reduce_scatter) {
              break;
            }
          // TODO: other compute ops
          } else if (is_slice_op(grad->producer()) && is_comm_op(grad->producer()->input(0)->producer())) {
            auto& comm_op_impl= reinterpret_cast<CommOpImpl&>(grad->producer()->input(0)->producer()->body());
            uint64_t comm_type = comm_op_impl.get_comm_type(grad->producer());
            if (comm_type != ALL_REDUCE_OP) {
              is_all_allreduce = false;
            }
            if (comm_type != REDUCE_SCATTER_OP) {
              is_all_reduce_scatter = false;
            }
            if (!is_all_allreduce && !is_all_reduce_scatter) {
              break;
            }
          } else {
            is_all_allreduce = false;
            is_all_reduce_scatter = false;
            break;
          }
        }
        if (is_all_allreduce || is_all_reduce_scatter) {
          is_need_sum_before_reduce = true;
        }
        multi_dst_ds.push_back(filtered[0]->get_distributed_states());
      }
      graph.CUR_STRATEGY_ID = 0;

      Tensor grad_sum;
      if (is_need_sum_before_reduce) {
        TensorList partial_grad_list;
        OpName name_of_sum_op_for = "";
        for (const auto& grad : filtered) {
          if (is_slice_op(grad->producer()) && is_comm_op(grad->producer()->input(0)->producer())) {
            ReplaceInput(grad->producer(), 0, grad->producer()->input(0)->producer()->input(0));
            Tensor partial_grad = grad;
            partial_grad_list.push_back(partial_grad);
            name_of_sum_op_for += (partial_grad->name() + ", ");
          } else {
            Tensor partial_grad = grad->producer()->input(0);
            partial_grad_list.push_back(partial_grad);
            name_of_sum_op_for += (partial_grad->name() + ", ");
          }
        }
        // if allreduce/reduce-scatter group is different between input grads,
        // then assert error in state deduce process.
        if (!is_fw_comm_op) {
          Tensor partial_grad_sum = MakeSumOp(partial_grad_list, OpMeta().set_name("sum_op_for_partial_[" + name_of_sum_op_for + "]"));
          partial_grad_sum->set_is_grad(true);
          partial_grad_sum->producer()->set_fw_op_id(fw_op_id);

          grad_sum = MakeCommOp(partial_grad_sum, multi_dst_ds, OpMeta().set_name("comm_op_after_partial_grad_sum"));
        } else {
          grad_sum = MakeSumOp(partial_grad_list, OpMeta().set_name("sum_op_for_partial_[" + name_of_sum_op_for + "]"));
        }
      } else {
        OpName name_of_sum_op_for = "";
        for (const auto& grad : filtered) {
          name_of_sum_op_for += (grad->name() + ", ");
        }
        grad_sum = MakeSumOp(filtered, OpMeta().set_name("sum_op_for_[" + name_of_sum_op_for + "]"));
      }
      grad_sum->set_is_grad(true);
      grad_sum->producer()->set_fw_op_id(fw_op_id);
      return grad_sum;
    }
  };

  // traverse the forward graph in the reversed topo order
  auto topo = Graph::TopoSort(ys, num_ops_hint);
  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {  
    auto& op = it->get();
    TensorList grad_outputs;
    if (op->num_outputs() > 0) {
      grad_outputs.reserve(op->num_outputs());
      for (auto& output : op->outputs()) {
        auto grad = reduce_grad(tensor_to_grads[output->id()], is_comm_op(op));
        tensor_to_reduced_grad[output->id()] = grad;
        grad_outputs.push_back(grad);
      }
    }
    if (op->num_inputs() > 0) {
      auto grad_inputs = op->Gradient(grad_outputs);
      for (size_t i = 0; i < grad_inputs.size(); i++) {
        if (!grad_inputs[i].is_defined())
          continue;
        
        grad_inputs[i]->set_is_grad(true);
        grad_inputs[i]->producer()->set_fw_op_id(op->id());

        // states deduce
        auto& grad_op = grad_inputs[i]->producer();
        Tensor final_grad = grad_inputs[i];
        // handle identify backward allreduce or varibale gradient all-reduce/reduce-scatter
        // other case should be handled by CommOp DoGradient
        // Workaround: for multi-task slice
        if (!(is_comm_op(op->input(i)->producer()) && op->input(i)->producer()->output(0)->num_consumers() == 1) && !is_slice_op(op->input(i)->producer())) {
          auto& graph = grad_inputs[i]->graph();
          DistributedStatesList multi_dst_ds;
          bool is_need_comm_op = false;
          for (size_t cur_strategy_id = 0; cur_strategy_id < graph.NUM_STRATEGY; cur_strategy_id++) {
            graph.CUR_STRATEGY_ID = cur_strategy_id;
            auto& ds_grad = grad_inputs[i]->get_distributed_states();
            if (ds_grad.is_valid()) {
              // do SplitAllReduce for multi-strategy
              if (is_variable_op(op->input(i)->producer()) && graph.NUM_STRATEGY > 1) {
                is_need_comm_op = true;
              }
              // HT_LOG_DEBUG << local_device << ": " << "grad_op: " << grad_op << ": states: " << ds_grad.ds_info() << ", shape: " << grad_inputs[i]->shape();
              if (ds_grad.get_dim(-2) > 1) { // partial->duplicate to sync the gradients for dp
                int32_t device_num = ds_grad.get_device_num();
                // std::pair<std::vector<int32_t>, int32_t> src2dst({{-2}, -1});
                std::pair<std::vector<int32_t>, int32_t> src2dst;
                if (is_variable_op(op->input(i)->producer()) && op->input(i)->get_distributed_states().zero()) {
                  // attention: the result tensor was dp grouped split0, not really split0!
                  // so should do allgather still within the same dp group later! 
                  src2dst = {{-2}, 0}; // reduce-scatter
                  // Question: consider exec graph switch
                  // for row parallel whose split is at dim 1
                  // maybe it's a better option to set src2dst = {{-2}, 1}
                } else {
                  src2dst = {{-2}, -1}; // allreduce
                }
                std::unordered_map<int32_t, int32_t> res_states = ds_grad.combine_states(src2dst);
                std::vector<int32_t> res_order = ds_grad.combine_order(src2dst);
                DistributedStates ds_dst({device_num, res_states, res_order});
                HT_LOG_TRACE << hetu::impl::comm::GetLocalDevice() << ": " 
                  << "backward: partial to duplicate: " << grad_inputs[i]
                  << ", src states: " << ds_grad.ds_info()
                  << ", dst states: " << ds_dst.ds_info();
                multi_dst_ds.push_back(ds_dst);
                is_need_comm_op = true;
              } else {
                multi_dst_ds.push_back(ds_grad);
              }
            } else {
              HT_LOG_ERROR << "ds_grad is invalid!";
            }
          }
          graph.CUR_STRATEGY_ID = 0;
          if (is_need_comm_op && !is_comm_op(op->input(i)->producer())) {
          // if (is_need_comm_op) {
            final_grad = MakeCommOp(grad_inputs[i], multi_dst_ds, 
              OpMeta().set_name("comm_op_after_" + grad_op->name())); // allreduce
            final_grad->set_is_grad(true);
            final_grad->producer()->set_fw_op_id(op->id());
          }
        }
        auto input = op->input(i);
        auto it = tensor_to_grads.find(input->id());
        if (it == tensor_to_grads.end())
          tensor_to_grads[input->id()] = {final_grad};
        else
          it->second.push_back(final_grad);
      }
    }

  }

  TensorList ret;
  ret.reserve(xs.size());
  for (auto& x : xs) {
    auto it = tensor_to_reduced_grad.find(x->id());
    if (it != tensor_to_reduced_grad.end())
      ret.push_back(it->second);
    else
      ret.emplace_back(Tensor());
  }
  return ret;
}

std::string GraphType2Str(GraphType type) {
  if (type == GraphType::EAGER) {
    return "eager";
  } else if (type == GraphType::DEFINE_BY_RUN) {
    return "define_by_run";
  } else if (type == GraphType::DEFINE_AND_RUN) {
    return "define_and_run";
  } else if (type == GraphType::EXECUTABLE) {
    return "executable";
  } else {
    HT_VALUE_ERROR << "Unrecognizable graph type: " << static_cast<int>(type);
    __builtin_unreachable();
  }
}

std::ostream& operator<<(std::ostream& os, GraphType type) {
  os << GraphType2Str(type);
  return os;
}

std::ostream& operator<<(std::ostream& os, const Graph& graph) {
  os << "graph(name=" << graph.name() << ", id=" << graph.id()
     << ", type=" << graph.type() << ")";
  return os;
}

} // namespace graph
} // namespace hetu
