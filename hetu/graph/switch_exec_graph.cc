#include "hetu/graph/headers.h"
#include "hetu/graph/graph.h"
#include "hetu/graph/executable_graph.h"
#include "hetu/graph/switch_exec_graph.h"
#include "hetu/graph/distributed_states.h"
#include "hetu/graph/operator.h"
#include "hetu/graph/ops/Split.h"
#include "hetu/graph/ops/Concatenate.h"
#include "hetu/graph/ops/Contiguous.h"
#include "hetu/graph/ops/placeholder.h"
#include "hetu/graph/ops/Communication.h"
#include "hetu/impl/communication/comm_group.h"
#include "hetu/impl/communication/mpi_comm_group.h"
#include "hetu/impl/communication/nccl_comm_group.h"
#include "hetu/impl/memory/CUDACachingMemoryPool.cuh"
#include "hetu/core/device.h"
#include "hetu/core/dtype.h"
#include "hetu/core/ndarray_meta.h"
#include "hetu/core/stream.h"
#include "hetu/common/timing.h"
#include <nvml.h>
#include <nccl.h>
#include <iostream>
#include <fstream>

namespace hetu {
namespace graph {

static std::unordered_map<DeviceGroup, std::once_flag> warmup_flags;
static std::unordered_map<std::pair<Device, Device>, P2PRoute> all_routes;

std::ostream& operator<<(std::ostream& os, const SwitchExecGraph& switcher) {
  os << "switch_exec_graph(" << switcher.SwitchGraphPair().first->name() << ", " 
    << switcher.SwitchGraphPair().second->name() << ")";
  return os;
}

template<typename Key, typename Value>
static std::unordered_set<Key> KeysUnion(const std::unordered_map<Key, Value>& map1, const std::unordered_map<Key, Value>& map2)
{
  std::unordered_set<Key> result;
  for (const auto& pair : map1) {
    result.insert(pair.first);
  }
  for (const auto& pair : map2) {
    result.insert(pair.first);
  }
  return result;
}

template<typename Key>
static std::unordered_set<Key> KeysUnion(const std::vector<Key>& vec1, const std::vector<Key>& vec2)
{
  std::unordered_set<Key> result;
  for (const auto& key : vec1) {
    result.insert(key);
  }
  for (const auto& key : vec2) {
    result.insert(key);
  }
  return result;
}

template<typename Key>
static std::unordered_set<Key> KeysUnion(const std::unordered_set<Key>& set1, const std::unordered_set<Key>& set2)
{
  std::unordered_set<Key> result;
  std::set_union(set1.begin(), set1.end(), set2.begin(), set2.end(), std::inserter(result, result.begin()));
  return result;
}

static const P2PRoute& GetP2PRoute(const Device& from, const Device& to) {
  auto p2p_pair = std::make_pair(from, to);
  auto it = all_routes.find(p2p_pair);
  if (it != all_routes.end()) {
    return it->second;
  }
  if (Device::compare_hostname(from, to) != 0) {
    all_routes[p2p_pair] = P2PRoute(P2P_ROUTE_LEVEL::NET);
  } else {
    // TODO: 非A100/A800
    all_routes[p2p_pair] = P2PRoute(P2P_ROUTE_LEVEL::NVLINK);
  }
  return all_routes[p2p_pair];
}

// nccl存在一些冷启动的开销
// 先简单地进行一个小的all-to-all
static void WarmUpComm(const std::unordered_set<Device>& comm_set) {
  auto local_device = hetu::impl::comm::GetLocalDevice();
  std::vector<int> ranks(comm_set.size());
  std::transform(comm_set.begin(), comm_set.end(), ranks.begin(), 
                 [&](const Device& device) { return hetu::impl::comm::DeviceToWorldRank(device); });
  std::sort(ranks.begin(), ranks.end());
  HT_LOG_DEBUG << "all-to-all warm up for " << comm_set
    << ", whose ranks after sort = " << ranks;
  auto& comm_group = hetu::impl::comm::NCCLCommunicationGroup::GetOrCreate(ranks, Stream(local_device, kSwitchCollectiveStream));
  auto data = NDArray::empty({comm_set.size()}, local_device, kFloat32, kSwitchCollectiveStream);
  comm_group->AlltoAll(data, data);
}

// 2024.1.20 deprecated: 对于那些concat的是所有本地slice的仍然需要buffer（因为现在所有param都做成buffer了）
// 递归查找来判断某一个concat算子是否需要concat buffer（涉及非本地的都需要）
static bool NeedConcatBuffer(const Operator& op) {
  if (is_batched_isend_irecv_op(op)) {
    return true;
  }
  for (const auto& input : op->inputs()) {
    if (NeedConcatBuffer(input->producer())) {
      return true;
    }
  }
  return false;
}

static void HandleConcatBuffer(const Tensor& tensor, TensorList& buffer) {
  auto local_device = hetu::impl::comm::GetLocalDevice();
  if (is_concat_op(tensor->producer())) {
    // concat原地啥都不干
    if (tensor->producer()->num_inputs() == 1) {
      HandleConcatBuffer(tensor->producer()->input(0), buffer);
    } 
    // concat发挥作用
    else {
      buffer.push_back(tensor);
      return;
    }
  } 
  // 即所有的concat都是原地啥都不干
  // 我们不得不通过插入contiguous算子的方式来强行转移buffer
  else {
    HT_ASSERT(is_batched_isend_irecv_op(tensor->producer()) || is_slice_op(tensor->producer()))
      << local_device << ": " << tensor->producer() << " type wrong";
    /*
    HT_LOG_INFO << local_device << ": " << tensor << " have " << tensor->num_consumers();
    for (auto& op_ref : tensor->consumers()) {
      auto& op = op_ref.get();
      HT_LOG_INFO << local_device << ": " << op << " ";
    }
    */
    auto new_tensor = MakeContiguousOp(tensor, 
                                       OpMeta().set_is_deduce_states(false).set_name("contiguous_" + tensor->name()));
    auto& contiguous_op = new_tensor->producer();
    if (tensor->placement() == local_device) {
      contiguous_op->MapToParallelDevices(tensor->placement_group());
      contiguous_op->Instantiate(local_device, kSwitchComputingStream);
      // 将contiguous算子插入到BatchedIsendIrecv/Slice算子和concat算子之间
      auto& consumer = tensor->consumer(0);
      HT_ASSERT(is_concat_op(consumer) && consumer->num_inputs() == 1)
        << "assumption error";
      Graph::ReplaceInput(consumer, 0, new_tensor);
      // contiguous算子输出记录在buffer中
      buffer.push_back(new_tensor);
    }
  }
}

void ParamBuffer::Alloc(const Stream& stream, 
                        bool use_nccl, 
                        ncclComm_t comm,
                        bool use_caching_mempool,
                        bool use_async) {
  TIK(alloc_time);
  auto local_device = hetu::impl::comm::GetLocalDevice(); 
  hetu::cuda::CUDADeviceGuard guard(local_device.index());
  HT_LOG_DEBUG << local_device << ": " << _name << " param buffer"
    << " will alloc " << (double)_buffer_size / (1024 * 1024) << " MiB";  
#if defined(NCCL_MAJOR) && defined(NCCL_MINOR) && (NCCL_MAJOR >= 2) && (NCCL_MINOR >= 19) 
  // HT_RUNTIME_ERROR << "NotImplementedError";
  if (use_nccl) {
    HT_ASSERT(comm != nullptr)
      << "nccl buffer registration must have a communicator";
    void* reg_handle;                                     
    ncclResult_t status = ncclMemAlloc(&_raw_ptr, _buffer_size);
    if (status != ncclSuccess) {
      HT_RUNTIME_ERROR << "ncclMemAlloc failed: " << ncclGetErrorString(status);
    }
    status = ncclCommRegister(comm, _raw_ptr, _buffer_size, &reg_handle);
    if (status != ncclSuccess) {
      HT_RUNTIME_ERROR << "ncclCommRegister failed: " << ncclGetErrorString(status);
    }         
    _storage = std::make_shared<NDArrayStorage>(BorrowToMemoryPool(
      local_device, _raw_ptr, _buffer_size, [=](DataPtr data_ptr) {
        TIK(free_time);
        hetu::cuda::CUDADeviceGuard guard(data_ptr.device.index());
        HT_LOG_DEBUG << local_device << ": " << _name << " param buffer"
          << " will free " << (double)_buffer_size / (1024 * 1024) << " MiB";  
        ncclResult_t status = ncclCommDeregister(comm, reg_handle);
        if (status != ncclSuccess) {
          HT_RUNTIME_ERROR << "ncclCommDeregister failed: " << ncclGetErrorString(status);
        }   
        status = ncclMemFree(data_ptr.ptr);
        if (status != ncclSuccess) {
          HT_RUNTIME_ERROR << "ncclMemFree failed: " << ncclGetErrorString(status);
        } 
        HT_LOG_DEBUG << local_device << ": " << _name << " param buffer free end";  
        TOK(free_time);
        _free_time = COST_MSEC(free_time);
      }));
  } else {
    if (use_caching_mempool) {
      if (!hetu::impl::AllocAfterFreeFromCUDACache(local_device, _raw_ptr, _buffer_size)) {
        HT_RUNTIME_ERROR << "cudaMalloc failed (OOM) "
          << "though releasing some data space from the caching mempool";
      }
    } else {
      cudaError_t status;
      if (!use_async || stream.is_blocking()) {
        status = cudaMalloc(&_raw_ptr, _buffer_size);
      } else {
        status = cudaMallocAsync(&_raw_ptr, _buffer_size, hetu::impl::CUDAStream(stream).cuda_stream());
      }
      if (status != cudaSuccess) {
        HT_RUNTIME_ERROR << "cudaMalloc failed: " << cudaGetErrorString(status);
      }
    }
    _storage = std::make_shared<NDArrayStorage>(BorrowToMemoryPool(
      local_device, _raw_ptr, _buffer_size, [=](DataPtr data_ptr) {
        TIK(free_time);
        hetu::cuda::CUDADeviceGuard guard(data_ptr.device.index());
        HT_LOG_DEBUG << local_device << ": " << _name << " param buffer"
          << " will free " << (double)_buffer_size / (1024 * 1024) << " MiB";  
        cudaError_t status;
        if (!use_async || stream.is_blocking()) {
          status = cudaFree(data_ptr.ptr);
        } else {
          status = cudaFreeAsync(data_ptr.ptr, hetu::impl::CUDAStream(stream).cuda_stream());
        }
        if (status != cudaSuccess) {
          HT_RUNTIME_ERROR << "cudaFree failed: " << cudaGetErrorString(status);
        }
        HT_LOG_DEBUG << local_device << ": " << _name << " param buffer free end";  
        TOK(free_time);
        _free_time = COST_MSEC(free_time);
      }));
  }
#else
  // Use AllocDataSpace will cause OOM
  /*
  // Note that we need to use kSwitchCollectiveStream for BufferBatchedIsendIrecv
  _storage = std::make_shared<NDArrayStorage>(AllocFromMemoryPool(local_device, _buffer_size, stream));
  _raw_ptr = _storage->mutable_data();
  */
  // Use BorrowDataSpace
  if (use_caching_mempool) {
    if (!hetu::impl::AllocAfterFreeFromCUDACache(local_device, _raw_ptr, _buffer_size)) {
      HT_RUNTIME_ERROR << "cudaMalloc failed (OOM) "
        << "though releasing some data space from the caching mempool";
    }
  } else {
    cudaError_t status;
    if (!use_async || stream.is_blocking()) {
      status = cudaMalloc(&_raw_ptr, _buffer_size);
    } else {
      status = cudaMallocAsync(&_raw_ptr, _buffer_size, hetu::impl::CUDAStream(stream).cuda_stream());
    }
    if (status != cudaSuccess) {
      HT_RUNTIME_ERROR << "cudaMalloc failed: " << cudaGetErrorString(status);
    }
  }
  _storage = std::make_shared<NDArrayStorage>(BorrowToMemoryPool(
    local_device, _raw_ptr, _buffer_size, [=](DataPtr data_ptr) {
      TIK(free_time);
      hetu::cuda::CUDADeviceGuard guard(data_ptr.device.index());
      HT_LOG_DEBUG << local_device << ": " << _name << " param buffer"
        << " will free " << (double)_buffer_size / (1024 * 1024) << " MiB";  
      cudaError_t status;
      if (!use_async || stream.is_blocking()) {
        status = cudaFree(data_ptr.ptr);
      } else {
        status = cudaFreeAsync(data_ptr.ptr, hetu::impl::CUDAStream(stream).cuda_stream());
      }
      if (status != cudaSuccess) {
        HT_RUNTIME_ERROR << "cudaFree failed: " << cudaGetErrorString(status);
      }
      HT_LOG_DEBUG << local_device << ": " << _name << " param buffer free end";  
      TOK(free_time);
      _free_time = COST_MSEC(free_time);
    }));
#endif
  _stream = stream;
  _is_allocated = true;
  TOK(alloc_time);
  _alloc_time = COST_MSEC(alloc_time);
  HT_LOG_DEBUG  << _name << " param buffer"
    << " alloc " << (double)_buffer_size / (1024 * 1024) << " MiB and cost " << _alloc_time << " ms";  
}

void ParamBuffer::Free() {
  _storage.reset();
  _raw_ptr = nullptr;
  _is_allocated = false;
  _is_auxiliary = false;
  HT_LOG_DEBUG << _name << " param buffer"
    << " free " << (double)_buffer_size / (1024 * 1024) << " MiB and cost " << _free_time << " ms";  
}

void ParamBuffer::Bind(const std::shared_ptr<NDArrayStorage>& storage) {
  auto local_device = hetu::impl::comm::GetLocalDevice();  
  HT_LOG_DEBUG << local_device << ": " << _name << " param buffer"
    << " will bind " << (double)_buffer_size / (1024 * 1024) << " MiB";  
  HT_ASSERT(!_is_allocated)
    << "ParamBuffer " << _name << " bind can only used when the storage is not allocated yet";
  HT_ASSERT(storage->device() == local_device)
    << "ParamBuffer " << _name << " device should equal to the storage device to bind"
    << ", but find ParamBuffer device " << local_device << " != storage device " << storage->device();
  HT_ASSERT(storage->size() == _buffer_size)
    << "ParamBuffer " << _name << " size should equal to the storage size to bind"
    << ", but find ParamBuffer size " << _buffer_size << " != storage size " << storage->size();
  _storage = storage;
  _raw_ptr = storage->mutable_data();
  _is_allocated = true;
  _is_auxiliary = true;
  HT_LOG_DEBUG << local_device << ": " << _name << " param buffer bind end"; 
}

size_t ParamBuffer::GetLayerwiseBucketId(const Tensor& tensor) {
  std::string sub_str = "Block";
  size_t pos = tensor->name().find(sub_str);
  HT_ASSERT (pos != std::string::npos) 
    << "Can't find block num in the tensor name " << tensor->name();
  size_t next_char_pos = pos + sub_str.length();
  HT_ASSERT (next_char_pos < tensor->name().length())
    << "Can't find block num in the tensor name " << tensor->name();
  std::string layer_num_str = "";
  while (tensor->name()[next_char_pos] != std::string::npos
         && tensor->name()[next_char_pos] >= '0' 
         && tensor->name()[next_char_pos] <= '9') {
    layer_num_str += tensor->name()[next_char_pos];
    next_char_pos += 1;
  }
  HT_ASSERT(layer_num_str != "")
    << "Cannot fetch the number after 'Block' for " << tensor->name();
  size_t layer_num = std::stoi(layer_num_str);
  return layer_num;
}

size_t ParamBuffer::GetCommGroupwiseBucketId(const Operator& update_op,
                                             const std::unordered_map<DeviceGroup, size_t>& comm_group_bucket_id_map) {
  HT_ASSERT(is_optimizer_update_op(update_op))
    << "update_op should be an optimizer update op, but got " << update_op;
  auto& grad = update_op->input(1);
  auto& grad_op = grad->producer();
  if (!is_grad_reduce_op(grad_op)) {
    return 0;
  }
  DeviceGroup comm_group;
  if (is_split_all_reduce_op(grad_op)) {
    auto& split_all_reduce_op_impl = dynamic_cast<SplitAllReduceOpImpl&>(grad_op->body());
    comm_group = split_all_reduce_op_impl.comm_groups()[0];
  } else if (is_all_reduce_op(grad_op)) {
    auto& all_reduce_op_impl = dynamic_cast<AllReduceOpImpl&>(grad_op->body());
    comm_group = all_reduce_op_impl.comm_group();
  }
  // size_t bucket_num = comm_group_list.size();
  size_t bucket_id = -1;
  auto it = comm_group_bucket_id_map.find(comm_group);
  if (it != comm_group_bucket_id_map.end()) {
    bucket_id = it->second;
  }
  // for (size_t i = 0; i < bucket_num; i++) {
  //   if (comm_group == comm_group_list[i]) {
  //     bucket_id = i;
  //     break;
  //   }
  // }
  HT_ASSERT(bucket_id != -1)
    << "Can't find bucket_id of grad_op " << grad_op << " with comm_group = " << comm_group;
  return bucket_id; 
}

void ParamSlice::AddOwnedSliceInst(const Device& device, const Tensor& tensor) {
  if (!_owned_slice_instances.empty()) {
    const HTShape& shape = _owned_slice_instances[0]->shape();
    const auto shape_size = shape.size();
    HT_ASSERT(shape_size == tensor->shape().size())
      << "the new slice instance shape should be equal to the old slice instance shape, " 
      << "but the new is " << tensor->shape() << " and the old is " << shape;
    for(size_t i = 0; i < shape_size; ++i) {
      HT_ASSERT(shape[i] == tensor->shape(i))
        << "the new slice instance shape should be equal to the old slice instance shape, "  
        << "but the new is " << tensor->shape() << " and the old is " << shape;
    }
  }
  _owned_devices.push_back(device);
  _owned_slice_instances.push_back(tensor);
  _switcher->RecordTensorInfo(tensor, name());
}

void ParamSlice::AddNeededSliceInst(const Device& device, const Tensor& tensor) {
  HT_ASSERT(!_owned_slice_instances.empty())
    << "the slice isn't owned by any devices, "
    << "please ensure you've added a slice instance before";
  const HTShape& shape = _owned_slice_instances[0]->shape();
  const auto shape_size = shape.size();
  HT_ASSERT(shape_size == tensor->shape().size())
    << "the needed slice shape should be equal to the owned slice shape, " 
    << "but the needed is " << tensor->shape() << " and the owned is " << shape;
  for(size_t i = 0; i < shape_size; ++i) {
    HT_ASSERT(shape[i] == tensor->shape(i))
      << "the needed slice shape should be equal to the owned slice shape, " 
      << "but the needed is " << tensor->shape() << " and the owned is " << shape;
  }
  _needed_devices.push_back(device);
  _needed_slice_instances.push_back(tensor);
  _switcher->RecordTensorInfo(tensor, name());
}

// TODO: 修改greedy算法
void ParamSlice::ParamSliceComm(Device2DTListPairMap& send_mapping,
                                Device2DTListPairMap& recv_mapping) {
  auto needed_len = _needed_slice_instances.size();
  auto owned_len = _owned_slice_instances.size();
  HT_ASSERT(needed_len == _needed_devices.size() && owned_len == _owned_devices.size())
    << "something wrong with the size";
  for (size_t i = 0; i < needed_len; ++i) {
    auto& needed_device = _needed_devices[i];
    bool already_owned = false;
    size_t already_owned_slice_instance_num = 0;
    // 先扫一遍，如果自己已经有了，那么就不需要通信了
    for (size_t j = 0; j < owned_len; ++j) {
      if (needed_device == _owned_devices[j]) {
        already_owned = true;
        already_owned_slice_instance_num= j;
        break;
      }
    }
    if (already_owned) {
      // 只需要替换即可
      // 不需要记录在send/recv的mapping中
      auto& old_tensor = _needed_slice_instances[i];
      auto& new_tensor = _owned_slice_instances[already_owned_slice_instance_num];
      HT_ASSERT(old_tensor->num_consumers() == 1)
        << "the slice instance should only used once (by a single concatenate op)";
      auto& consumer = old_tensor->consumer(0);
      for (size_t j = 0; j < consumer->num_inputs(); ++j) {
        if (consumer->input(j)->id() == old_tensor->id()) {
          Graph::ReplaceInput(consumer, j, new_tensor);
        }
      }
      HT_LOG_DEBUG_IF(needed_device == hetu::impl::comm::GetLocalDevice())
        << needed_device << ": can reuse the " << name()
        << " param slice instance owned by itself";
    } else {
      // 需要通信
      // 通信关系会记录在send/recv的mapping中
      Tensor send_tensor; // TBD
      Device send_device; // TBD
      auto& recv_tensor = _needed_slice_instances[i];
      auto& recv_device = _needed_devices[i];
      // 不同的算法
      // FCFS or round-robin
      if (_switcher->_algorithm_level == SWITCH_ALGORITHM_LEVEL::FCFS
          || _switcher->_algorithm_level == SWITCH_ALGORITHM_LEVEL::ROUND_ROBIN
          || _switcher->_algorithm_level == SWITCH_ALGORITHM_LEVEL::MULTI_NODE_ROUND_ROBIN) {
        send_tensor = _owned_slice_instances[_round_robin];
        send_device = _owned_devices[_round_robin];
        // 更新轮询次数
        // 多node情形下要额外考虑跨机间通信
        // 尽可能将其避免（除非避免不了）
        if (_switcher->_algorithm_level == SWITCH_ALGORITHM_LEVEL::MULTI_NODE_ROUND_ROBIN) {
          size_t round = 0, max_round = _owned_slice_instances.size();
          while (round < max_round) {
            send_tensor = _owned_slice_instances[(_round_robin + round) % max_round];
            send_device = _owned_devices[(_round_robin + round) % max_round];
            const auto& p2p_route = GetP2PRoute(send_device, recv_device);
            if (p2p_route.route_level() >= P2P_ROUTE_LEVEL::NET) {
              round++;
            } else {
              break;
            }
          }
        }
        if (_switcher->_algorithm_level == SWITCH_ALGORITHM_LEVEL::ROUND_ROBIN
            || _switcher->_algorithm_level == SWITCH_ALGORITHM_LEVEL::MULTI_NODE_ROUND_ROBIN) {
          _round_robin++;
        }
        if (_round_robin == _owned_slice_instances.size()) {
          _round_robin = 0;
        }
      } 
      // 按照已经通信的次数进行greedy（即选取已通信中p2p次数最小的）
      if (_switcher->_algorithm_level == SWITCH_ALGORITHM_LEVEL::GREEDY) {
        std::pair<Device, Device> best_p2p;
        size_t best_send_num;
        auto& p2p_val_mapping = _switcher->_p2p_val_mapping;
        size_t min_val = std::numeric_limits<size_t>::max();
        for (size_t j = 0; j < owned_len; ++j) {
          // p2p是双向通路
          // 规定device编号小的在前
          std::pair<Device, Device> p2p;
          if (_owned_devices[j] < recv_device) {
            p2p = std::make_pair(_owned_devices[j], recv_device);
          } else {
            p2p = std::make_pair(recv_device, _owned_devices[j]);
          }
          auto it = p2p_val_mapping.find(p2p);
          // 相当于通信了0次
          if (it == p2p_val_mapping.end()) {
            best_p2p = p2p;
            best_send_num = j;
            break;
          }
          // 选择通信次数最小的
          if (it->second < min_val) {
            min_val = it->second;
            best_p2p = p2p;
            best_send_num = j;
          }
        }
        // 更新p2p_val_mapping
        p2p_val_mapping[best_p2p] += 1;
        send_tensor = _owned_slice_instances[best_send_num];
        send_device = _owned_devices[best_send_num];
      }
      // 2024.2.25
      // 修正版greedy
      if (_switcher->_algorithm_level == SWITCH_ALGORITHM_LEVEL::NEW_GREEDY) {
        size_t best_send_num, round = 0;
        auto& intra_device_val_mapping = _switcher->_intra_device_val_mapping;
        auto& inter_device_val_mapping = _switcher->_inter_device_val_mapping;
        size_t min_val = std::numeric_limits<size_t>::max();
        for (size_t j = 0; j < owned_len; ++j) {
          const auto& p2p_route = GetP2PRoute(_owned_devices[j], recv_device);
          // 能不跨机，就不跨机
          if (p2p_route.route_level() >= P2P_ROUTE_LEVEL::NET) {
            round++;
            continue;
          }
          auto it = intra_device_val_mapping.find(_owned_devices[j]);
          // 相当于通信为0
          if (it == intra_device_val_mapping.end()) {
            best_send_num = j;
            break;
          }
          // 选择通信量除以带宽最小的
          if (it->second < min_val) {
            min_val = it->second;
            best_send_num = j;
          }
        }
        // 如果不得不跨机，那么再扫一遍
        if (round == owned_len) {
          for (size_t j = 0; j < owned_len; ++j) {
            auto it = inter_device_val_mapping.find(_owned_devices[j]);
            // 相当于通信为0
            if (it == inter_device_val_mapping.end()) {
              best_send_num = j;
              break;
            }
            // 选择通信量除以带宽最小的
            if (it->second < min_val) {
              min_val = it->second;
              best_send_num = j;
            }
          }
        }
        // 更新device_val_mapping
        const auto& p2p_route = GetP2PRoute(_owned_devices[best_send_num], recv_device);
        if (p2p_route.route_level() >= P2P_ROUTE_LEVEL::NET) {
          inter_device_val_mapping[_owned_devices[best_send_num]] += numel();
        } else {
          intra_device_val_mapping[_owned_devices[best_send_num]] += numel();
        }
        send_tensor = _owned_slice_instances[best_send_num];
        send_device = _owned_devices[best_send_num];
      }
      // 建立通信关系
      auto recv_it = recv_mapping.find(recv_device);
      auto send_it = send_mapping.find(send_device);
      HT_ASSERT(send_it != send_mapping.end() && recv_it != recv_mapping.end())
        << "device is not recorded in the send/recv mapping";
      recv_it->second.first.push_back(send_device);
      recv_it->second.second.push_back(recv_tensor);
      send_it->second.first.push_back(recv_device);
      send_it->second.second.push_back(send_tensor);
      HT_LOG_DEBUG_IF(send_device == hetu::impl::comm::GetLocalDevice())
        << send_device << ": will send the " << name()
        << " param slice instance to " << recv_device;
    }
  }
}

// 遍历ParamBlock中的每个ParamSlice
// 找到最优的ParamSliceInst的通信策略
void ParamBlock::ParamBlockComm(Device2DTListPairMap& send_mapping,
                                Device2DTListPairMap& recv_mapping) {
  // auto param_slices_size = _param_slices.size();
  for (auto& param_slice_ptr : _param_slices) {
    param_slice_ptr->ParamSliceComm(send_mapping, recv_mapping);
  }
}

// 递归地为ParamBlock创建所有的ParamSlices
void SwitchExecGraph::CreateParamBlock(ParamBlock& block,
                                      std::vector<int32_t>& slice_num, 
                                      const TensorName& block_name,
                                      int32_t dim) {
  const auto& block_shape = block.BlockShape();
  if (dim == block_shape.size()) {
    block.GetParamSlices().emplace_back(std::make_shared<ParamSlice>(block_name,
                                                                     block.SliceShape(),
                                                                     slice_num, 
                                                                     this));
    return;
  }
  for (int32_t i = 0; i < block_shape[dim]; ++i) {
    slice_num[dim] = i;
    CreateParamBlock(block, slice_num, block_name, dim + 1);
  }
}

// 作为发送端
// 将ParamBlock划分成OwnedParamSlice（抽象）
// 切分param成ParamSliceInstance（实际的tensor）
void SwitchExecGraph::MakeAllParamSlices(const Tensor& param, ParamBlock& block, 
                                    const Device& device, const DeviceGroup& group,
                                    std::vector<int32_t>& slice_num, std::vector<int32_t>& slice_relative_num,
                                    const std::unordered_map<int32_t, int32_t>& state,
                                    const std::vector<int32_t>& multiple, int32_t dim) {
  if (dim == multiple.size()) {
    auto& param_slice = block.GetParamSlice(slice_num);
    HTShape indices(slice_relative_num.begin(), slice_relative_num.end()); // int32_t -> int64_t
    HTShape splits(multiple.begin(), multiple.end()); // int32_t -> int64_t
    // 都先进行split
    auto split_output = MakeSplitOp(param, 
                                    indices, 
                                    splits, 
                                    OpMeta().set_name(param_slice->name()).set_is_deduce_states(false));
    auto& split_op = split_output->producer();
    // 其他device上生成的不需要map placement_group和placement
    if (hetu::impl::comm::GetLocalDevice() == device) { 
      split_op->MapToParallelDevices(group);
      split_op->Instantiate(device, kSwitchComputingStream);
    }
    // dup会导致一个param_slice对应多个slice_instance
    // 这也是这个优化问题之所以这么复杂的原因
    param_slice->AddOwnedSliceInst(device, std::move(split_output));
    return;
  } 
  int32_t basic_slice_num = 0;
  auto it = state.find(dim);
  if (it != state.end()) {
    basic_slice_num = it->second * multiple[dim];
  }
  for (int32_t i = 0; i < multiple[dim]; ++i) {
    slice_num[dim] = basic_slice_num + i;
    slice_relative_num[dim] = i;
    MakeAllParamSlices(param, block, device, group, slice_num, slice_relative_num, state, multiple, dim + 1);
  }                            
}

// 作为接收端
// 将ParamBlock划分成NeededParamSlice（抽象）
// 合并ParamSliceInstance成param（实际的tensor）
Tensor SwitchExecGraph::MergeAllParamSlices(const Tensor& param, ParamBlock& block, 
                                    const Device& device, const DeviceGroup& group,
                                    std::vector<int32_t>& slice_num, std::vector<int32_t>& slice_relative_num,
                                    const std::unordered_map<int32_t, int32_t>& state,
                                    const std::vector<int32_t>& multiple, int32_t dim) {
  if (dim == multiple.size()) {
    auto& param_slice = block.GetParamSlice(slice_num);
    const auto& owned_slice_instance = param_slice->OwnedSliceInst(0);
    // 之后会被替换成BatchedISendIRecvOp算子
    // Question: MakePlaceholderOp的各个参数是否都有必要
    auto needed_slice_instance = MakePlaceholderOp(owned_slice_instance->meta(), 
                                                   param->get_distributed_states(), 
                                                   OpMeta().set_device_groups({group}));
    param_slice->AddNeededSliceInst(device, needed_slice_instance);
    return needed_slice_instance;
  } 
  int32_t basic_slice_num = 0;
  TensorList merged_slices;
  auto it = state.find(dim);
  if (it != state.end()) {
    basic_slice_num = it->second * multiple[dim];
  }
  for (int32_t i = 0; i < multiple[dim]; ++i) {
    slice_num[dim] = basic_slice_num + i;
    slice_relative_num[dim] = i;
    Tensor merged_slice = MergeAllParamSlices(param, block, device, group, slice_num, slice_relative_num, state, multiple, dim + 1);
    merged_slices.push_back(std::move(merged_slice));
  }  
  auto concatenate_name = "concat_" + param->name() + "_at_dim_" + std::to_string(dim);
  auto concatenate_output = MakeConcatenateOp(std::move(merged_slices), dim, 
                                              OpMeta().set_name(concatenate_name).set_is_deduce_states(false));         
  auto& concatenate_op = concatenate_output->producer();
  // 其他device上生成的不需要map placement_group和placement
  if (hetu::impl::comm::GetLocalDevice() == device) { 
    concatenate_op->MapToParallelDevices(group);
    concatenate_op->Instantiate(device, kSwitchComputingStream);
  }  
  return concatenate_output;
}

// 对于一个param
// 进行全局的switch
// 每台机器会知道自己拥有这个param的哪些部分以及需要这个param的哪些部分
void SwitchExecGraph::SwitchParam(const DistributedStates& src_ds, const DeviceGroup& src_group,
                                   const DistributedStates& dst_ds, const DeviceGroup& dst_group,
                                   const Tensor& comm_input, const Tensor& after_param) {
  auto local_device = hetu::impl::comm::GetLocalDevice();
  std::vector<int32_t> block_shape;
  HTShape slice_shape;
  std::vector<int32_t> src_multiple;
  std::vector<int32_t> dst_multiple;
  // 获得最小粒度的块划分
  int32_t param_dims = comm_input->global_shape().size(); // size_t -> int32_t
  for (int32_t key = -2; key < param_dims; ++key) {
    auto src_dim = src_ds.states(key);
    auto dst_dim = dst_ds.states(key);
    if (key == -2) {
      HT_ASSERT(src_dim == 1 && dst_dim == 1) 
        << "parameter ds shouldn't have partial dim";
      continue;
    }
    if (key == -1) {
      continue;
    }
    auto max_dim = std::max(src_dim, dst_dim);
    HT_ASSERT(max_dim % src_dim == 0 && max_dim % dst_dim == 0)
      << "only support scaling by an integer";
    block_shape.push_back(max_dim); 
    slice_shape.push_back(comm_input->global_shape().at(key) / max_dim);
    src_multiple.push_back(max_dim / src_dim);
    dst_multiple.push_back(max_dim / dst_dim);
  }
  // 为当前param创建一个全局的、抽象的ParamBlock
  // 并为每个最小粒度的块划分创建一个抽象的ParamSlice
  // 其只需要知道ds的切分shape，而不需要绑定param的真实shape
  const TensorName& param_block_name = after_param->name();
  HT_LOG_DEBUG << local_device << ": make an abstract block for " << param_block_name
    << ", whose shape is " << block_shape;
  auto param_block_ptr = std::make_shared<ParamBlock>(param_block_name, block_shape, slice_shape, this);
  std::vector<int32_t> slice_num(param_dims, 0);
  CreateParamBlock(*param_block_ptr, slice_num, param_block_name, 0);
  _param_blocks.push_back(param_block_ptr);
  // 每个device作为发送端
  // 求出每个device拥有的小块儿并进行切分
  auto src_devices_size = src_group.num_devices();
  for(size_t i = 0; i < src_devices_size; ++i) {
    // 初始化发送端的mapping
    auto it = _send_mapping.find(src_group.get(i));
    if (it == _send_mapping.end()) {
      _send_mapping[src_group.get(i)] = std::make_pair(std::vector<Device>{}, std::vector<Tensor>{});
    }
    auto cur_state_index = src_ds.map_device_to_state_index(i);
    std::vector<int32_t> cur_slice_num(param_dims, 0);
    std::vector<int32_t> cur_slice_relative_num(param_dims, 0);
    // 进行具体的切分
    // 将ParamSliceInstance放入对应的ParamSlice
    HT_LOG_DEBUG << local_device << ": MakeAllParamSlices for tensor " << comm_input << " at device " << src_group.get(i)
      << ", cur_state_index = " << cur_state_index << " and src_multiple = " << src_multiple;
    MakeAllParamSlices(comm_input, *param_block_ptr, src_group.get(i), src_group, cur_slice_num, cur_slice_relative_num, 
                       cur_state_index, src_multiple, 0);
  }
  // 每个device作为接收端
  // 求出每个device需要的小块儿并进行合并
  auto dst_devices_size = dst_group.num_devices();
  for(size_t i = 0; i < dst_devices_size; ++i) {
    // 初始化发送端的mapping
    auto it = _recv_mapping.find(dst_group.get(i));
    if (it == _recv_mapping.end()) {
      _recv_mapping[dst_group.get(i)] = std::make_pair(std::vector<Device>{}, std::vector<Tensor>{});
    }
    auto cur_state_index = dst_ds.map_device_to_state_index(i);
    std::vector<int32_t> cur_slice_num(param_dims, 0);
    std::vector<int32_t> cur_slice_relative_num(param_dims, 0);
    // 进行具体的合并
    // 将新的ParamSliceInstance放入对应的ParamSlice
    // 会先用placeholder（之后再用BatchedISendIRecvOp进行替换）表征ParamSliceInstance
    // 返回的result即为新exec graph中最终合并后的param
    HT_LOG_DEBUG << local_device << ": MergeAllParamSlices for tensor " << after_param << " at device " << dst_group.get(i)
      << ", cur_state_index = " << cur_state_index << " and dst_multiple = " << dst_multiple;
    auto result = MergeAllParamSlices(after_param, *param_block_ptr, dst_group.get(i), dst_group, cur_slice_num, cur_slice_relative_num, 
                                      cur_state_index, dst_multiple, 0);
    // 如果是local的result
    // 记录result以及其与after graph param的映射
    if (local_device == dst_group.get(i)) {
      _comm_results_mapping.insert(std::make_pair(result->id(), after_param));
      _comm_results.push_back(std::move(result));
    }
  }
}

void SwitchExecGraph::MakeCommGraph(SWITCH_MODE switch_mode, SWITCH_LEVEL switch_level) {

  auto local_device = hetu::impl::comm::GetLocalDevice();
  HT_LOG_DEBUG << local_device << ": make a new comm graph begin...";

  auto& before_graph = _switch_graph_pair.first;
  auto& after_graph = _switch_graph_pair.second;
  auto& before_mapping = _define_graph->GetPlan(_switch_plan_pair.first).tensor_to_exec_tensor_mapping;
  auto& after_mapping = _define_graph->GetPlan(_switch_plan_pair.second).tensor_to_exec_tensor_mapping;
  auto& before_transfer_map = before_graph->_transfer_map;
  auto& after_transfer_map = after_graph->_transfer_map;
  auto& before_grad_map = before_graph->_grad_map;
  auto& after_grad_map = after_graph->_grad_map;

  std::string comm_graph_name_prefix;
  if (switch_mode == SWITCH_MODE::SWITCH_ORIGIN_PARAM
      || switch_mode == SWITCH_MODE::SWITCH_TRANSFER_PARAM) {
    comm_graph_name_prefix = "param";
  } else if (switch_mode == SWITCH_MODE::SWITCH_CURRENT_GRAD
             || switch_mode == SWITCH_MODE::SWITCH_ACCUMULATE_GRAD) {
    comm_graph_name_prefix = "grad";
  } else {
    HT_RUNTIME_ERROR << "switch mode type wrong";
  }
  _comm_graph = Graph::_make_new_graph<ExecutableGraph>(
    comm_graph_name_prefix + "_comm_graph_between_" + before_graph->name() 
    + "_and_" + after_graph->name());

  Graph::push_graph_ctx(_comm_graph->id());
  
  std::unordered_set<Device> src_set;
  std::unordered_set<Device> dst_set;
  DataType dtype = DataType::UNDETERMINED;
  for (auto& define_param_ref : _define_graph_params) {
    auto& define_param = define_param_ref.get();
    // Test Case
    /*
    if ("wte_table" != define_param->name()) {
      continue;
    }
    */
    auto define_param_id = define_param->id();
    auto before_it = before_mapping.find(define_param_id);
    bool is_before_active = before_it != before_mapping.end();
    auto after_it = after_mapping.find(define_param_id);
    bool is_after_active = after_it != after_mapping.end();
    // AMP情形需要使用transfer param而不是origin param
    // 这里直接修正before_it和after_it即可
    // Question: transfer param作为DataTransferOp是否会和ParallelVariableOp有所不同
    if (switch_mode == SWITCH_MODE::SWITCH_TRANSFER_PARAM) {
      if (is_before_active) {
        before_it = before_transfer_map.find(before_it->second->id());
        HT_ASSERT(before_it != before_transfer_map.end())
          << "before transfer map dose not consist of " << before_it->second;
      }
      if (is_after_active) {
        after_it = after_transfer_map.find(after_it->second->id());
        HT_ASSERT(after_it != after_transfer_map.end())
          << "after transfer map dose not consist of " << after_it->second;
      }
    }
    // 切换grad的情形则把grad来当作param处理
    if (switch_mode == SWITCH_MODE::SWITCH_CURRENT_GRAD
        || switch_mode == SWITCH_MODE::SWITCH_ACCUMULATE_GRAD) {
      if (is_before_active) {
        before_it = before_grad_map.find(before_it->second->id());
        HT_ASSERT(before_it != before_grad_map.end())
          << "before grad map dose not consist of " << before_it->second;
      }
      if (is_after_active) {
        after_it = after_grad_map.find(after_it->second->id());
        HT_ASSERT(after_it != after_grad_map.end())
          << "after grad map dose not consist of " << after_it->second;
      }
    }
    // 分情况讨论
    HT_LOG_DEBUG << local_device << ": processing param " << define_param << " in switch from "
      << before_graph->name() << " to " << after_graph->name();
    // 两个都不在
    if (!is_before_active && !is_after_active) {
      // 尽管在define and run graph中创建了某一param
      // 但在实际的exec graph中并没有进行使用
      // 例如lm_head_weight
      // 这种情况我们什么都不处理
      HT_LOG_DEBUG << local_device << ": param " << define_param << " is not in both graphs";
    } 
    // 只在前头的图里
    else if (is_before_active && !is_after_active) {
      // TODO: save the param back to the cpu
      HT_LOG_DEBUG << local_device << ": param " << define_param << " is only in the before graph";
      auto& before_param = before_it->second;
      HT_RUNTIME_ERROR << "NotImplementedError";
    } 
    // 只在后头的图里
    else if (!is_before_active && is_after_active) {
      if (switch_mode == SWITCH_MODE::SWITCH_ORIGIN_PARAM) {
        // 为了保证正确性我们这里还是会从add init里再度对其赋值
        // 这里只对after的add init赋值而不会产生性能上的开销
        // 目的是为了防止在新的after graph中有新的topo而需要访问这一before graph中未使用的param
        HT_LOG_DEBUG << local_device << ": param " << define_param << " is only in the after graph";
        auto& after_param = after_it->second;
        auto add_on_inits_it = _define_graph->_add_on_inits.find(define_param->id());
        if (add_on_inits_it != _define_graph->_add_on_inits.end()) {
          HT_LOG_DEBUG << local_device << ": param " << define_param << " in the after graph is reset";
          Graph::ResetVariableData(after_param, *add_on_inits_it->second);
        } else {
          // 另外一种情况是param不在_add_on_inits里
          // 即没有被修改过provided data
          // 这种情况不需要handle（exec graph的AllocVariableDataInner会自动帮忙处理）
          HT_LOG_DEBUG << local_device << ": param " << define_param << " in the after graph will be lazily initialized";
        }
      }
      else {
        HT_RUNTIME_ERROR << "NotImplementedError";
      }
    } 
    // 两个图都在（这种情况才是我们核心要考虑的）
    else {
      HT_LOG_DEBUG << local_device << ": param " << define_param << " is in both graphs";
      // 注意这里用了一个trick
      // 如果是param那么这里的iter都是tensor_to_exec_tensor_mapping的iter
      // 如果是grad那么这里的iter都是grad_map的iter
      auto& before_param = before_it->second;
      auto& after_param = after_it->second;
      HT_ASSERT(before_param->global_shape() == after_param->global_shape())
        << "parameter shapes in the two switching exec graphs should be equal"
        << ", but find global shape of " << before_param << " in before graph is " << before_param->global_shape()
        << " and global shape of " << after_param << " in after graph is " << after_param->global_shape();
      // 确定dtype
      HT_ASSERT(before_param->dtype() == after_param->dtype())
        << local_device << ": before param " << before_param << " dtype is " << before_param->dtype()
        << " and after param " << after_param << " dtype is " << after_param->dtype()
        << ", they should be equal";
      if (dtype == DataType::UNDETERMINED) {
        dtype = before_param->dtype();
      } else {
        HT_ASSERT(dtype == before_param->dtype())
          << "we only support homogeneous dtype now, but there are two param dtypes: "
          << dtype << " and " << before_param->dtype();
      }    
      // 确定ds和device group  
      const auto& src_ds = before_param->get_distributed_states();
      const auto& src_group = before_param->producer()->device_group();
      const auto& dst_ds = after_param->get_distributed_states();
      const auto& dst_group = after_param->producer()->device_group();
      HT_ASSERT(!src_group.empty() && !dst_group.empty())
        << "src group and dst group shouldn't be empty, but before param "
        << before_param << " src_group is " << src_group << " and after param "
        << after_param << " dst group is " << dst_group;
      // 将出现过的device都放入comm set中
      for (auto& device : src_group.devices()) {
        if (src_set.find(device) == src_set.end()) {
          src_set.insert(device);
        }
        // 用来之后BatchedIsendIrecv以及MPI同步的
        if (_comm_set.find(device) == _comm_set.end()) {
          _comm_set.insert(device);
        }
      }
      for (auto& device : dst_group.devices()) {
        if (dst_set.find(device) == dst_set.end()) {
          dst_set.insert(device);
        }
        // 用来之后BatchedIsendIrecv以及MPI同步的
        if (_comm_set.find(device) == _comm_set.end()) {
          _comm_set.insert(device);
        }
      }
      // 依据before_param生成通信图input的placeholder以及相应的feed_dict
      // 理论上也可以直接使用before_param作为input的tensor
      // 但这里还是希望尽量保证comm graph与before graph之间的隔离性
      auto comm_input = MakePlaceholderOp(before_param->meta(), 
                                          before_param->get_distributed_states(), 
                                          OpMeta().set_device_groups({src_group}).set_name(before_param->name() + "_comm_input"));
      if (src_group.contains(local_device)) {
        comm_input->producer()->MapToParallelDevices(src_group);
        comm_input->producer()->Instantiate(local_device, kSwitchComputingStream);
        // 只求comm graph的topo的话并不需要实际的feed dict
        if (switch_level != SWITCH_LEVEL::TOPO) {
          auto comm_input_data_it = before_graph->_preserved_data.find(before_param->id());
          HT_ASSERT(comm_input_data_it != before_graph->_preserved_data.end())
            << "something wrong, the data of " << before_param << " to switch in the before graph is not available";
          HT_ASSERT(comm_input->shape() == comm_input_data_it->second->shape())
            << "comm graph placeholder shape " << comm_input->shape() << " should be equal to"
            << " the shape of the preserved data " << comm_input_data_it->second->shape();
          _comm_feed_dict.insert(std::make_pair(comm_input->id(), NDArrayList({comm_input_data_it->second})));
        }    
        _comm_feed_dict_mapping.insert(std::make_pair(comm_input->id(), before_param));
      }
      // 生成该param切分和合并的计算图
      // 并建立映射关系
      // 只是知道哪些device需要哪些slice
      // 哪些device拥有哪些slice
      // 不进行实际的算法决策
      HT_LOG_DEBUG << local_device << ": switch param from " << before_param << " to " << after_param
        << ", src group = " << src_group << " and dst_group = " << dst_group 
        << ", src ds states = " << src_ds.get_states() << " and dst states = " << dst_ds.get_states();
      SwitchParam(src_ds, src_group, dst_ds, dst_group, comm_input, after_param);
    }
  }

  // 从全局的ParamBlocks视角出发
  // 选择最优的通信方案
  // 目前最优的是对于每一个ParamBlock的每一个ParamSlice，采用round-robin的算法
  for (auto& param_block_ptr : _param_blocks) {
    param_block_ptr->ParamBlockComm(_send_mapping, _recv_mapping);
  }

  // _send_mapping和_recv_mapping此时已经获取到所有params的通信方案
  // 将中间的placeholder算子替换为具体的通信算子
  HT_LOG_DEBUG << local_device << ": make the crucial BatchedISendIRecvOp begin...";
  std::vector<Device> src_devices(src_set.begin(), src_set.end());
  std::vector<Device> dst_devices(dst_set.begin(), dst_set.end());
  std::vector<Device> comm_devices(_comm_set.begin(), _comm_set.end());
  // local_device is exclusive
  auto comm_device_group = DeviceGroup(comm_devices);
  if (!comm_device_group.contains(local_device)) {
    HT_LOG_DEBUG << local_device << ": no params can leverage hot switch";
    Graph::pop_graph_ctx();
    HT_LOG_DEBUG << local_device << ": make a new comm graph end...";
    return;
  }
  // local_device send to other devices
  std::vector<Device>& send_to_devices = _send_mapping[local_device].first;
  TensorList& send_tensors = _send_mapping[local_device].second;
  auto send_len = send_tensors.size();
  // local_device receive from other devices
  std::vector<Device>& recv_from_devices = _recv_mapping[local_device].first;
  HTShapeList recv_tensor_shapes;
  auto recv_len = _recv_mapping[local_device].second.size();
  for (size_t i = 0; i < recv_len; ++i) {
    recv_tensor_shapes.push_back(_recv_mapping[local_device].second[i]->shape());
  }
  // BatchedISendIRecv Part
  HT_LOG_DEBUG << local_device << ": will send " << send_len << " tensor to device " 
    << send_to_devices << " and recv " << recv_len << " tensor from other devices"
    << ", the src devices = " << src_devices << " and comm devices = " << comm_devices;
  // 作为发送端设置多条发送的buffer
  // TensorList contiguous_send_tensors;
  // contiguous_send_tensors.reserve(send_tensors.size());
  for (size_t i = 0; i < send_len; ++i) {
    // 我们这里不再单独插入contiguous算子
    // 而是在处理ParamBuffer时统一进行contiguous操作
    /*
    // 在通信前插入contiguous算子
    // profile时单独计时
    auto& send_tensor = send_tensors[i];
    auto contiguous_send_tensor = MakeContiguousOp(send_tensor, 
                                                   OpMeta().set_is_deduce_states(false));
    auto& contiguous_op = contiguous_send_tensor->producer();
    HT_ASSERT(send_tensor->placement_group().contains(local_device))
      << "send tensor should already be instantiated locally";
    contiguous_op->MapToParallelDevices(send_tensor->placement_group());
    contiguous_op->Instantiate(local_device, kSwitchComputingStream);
    contiguous_send_tensors.push_back(std::move(contiguous_send_tensor));
    */
    // 给所有发向同一个device的tensor记录一个ParamBuffer
    // 之后通信时会发送一整个buffer
    // 注意这里记录的都是未进行contiguous的tensor
    auto it = _send_buffers.find(send_to_devices[i]);
    if (it == _send_buffers.end()) {
      _send_buffers[send_to_devices[i]] = std::make_shared<ParamBuffer>("send_" + std::to_string(send_to_devices[i].index()), 
                                                                        TensorList{send_tensors[i]});
    } else {
      _send_buffers[send_to_devices[i]]->AddTensor(send_tensors[i]);
    }
  }
  // 这里只是记录一个BatchedISendIRecvOp
  // 后续的实现其实是使用ParamBuffer
  auto result = MakeBatchedISendIRecvOp(send_tensors, send_to_devices, 
                                        recv_tensor_shapes, recv_from_devices, 
                                        comm_devices, dtype, 
                                        OpMeta().set_is_deduce_states(false));
  auto& batched_isend_irecv_op = result->producer();
  batched_isend_irecv_op->MapToParallelDevices(comm_device_group);
  batched_isend_irecv_op->Instantiate(local_device, kSwitchCollectiveStream);
  TensorList recv_tensors = batched_isend_irecv_op->outputs();
  // we need to add dummy link for topo sort
  // 只有send没有recv
  // 要将这种情况的dummy link放到fetch中
  if (recv_len == 0) {
    HT_LOG_DEBUG << local_device << ": no recv from other devices";
    HT_ASSERT(result == batched_isend_irecv_op->out_dep_linker())
      << "something wrong, it should be the out_dep_linker";
    _dummy_links.push_back(result);
  } else {
    HT_LOG_DEBUG << local_device << ": recv from devices " << recv_from_devices;
    // 作为接收端设置多条接收的buffer
    for (size_t i = 0; i < recv_len; ++i) {
      auto it = _recv_buffers.find(recv_from_devices[i]);
      if (it == _recv_buffers.end()) {
        _recv_buffers[recv_from_devices[i]] = std::make_shared<ParamBuffer>("recv_" + std::to_string(recv_from_devices[i].index()), 
                                                                            TensorList{recv_tensors[i]});
      } else {
        _recv_buffers[recv_from_devices[i]]->AddTensor(recv_tensors[i]);
      }
    }
  }
  HT_LOG_DEBUG << local_device << ": make the crucial " << result << " end..";

  // 将原先的placeholder替换为recv_tensor
  HT_ASSERT(recv_len == recv_tensors.size())
    << "something wrong with the recv len";
  for (size_t i = 0; i < recv_len; ++i) {
    auto& old_tensor = _recv_mapping[local_device].second[i];
    auto& new_tensor = recv_tensors[i];
    HT_ASSERT(old_tensor->num_consumers() == 1)
      << "the slice instance should only used once (by a single concatenate op)";
    auto it = _info_mapping.find(old_tensor->id());
    HT_ASSERT(it != _info_mapping.end())
      << "the info of the old tensor is not recorded";
    RecordTensorInfo(new_tensor, it->second);
    auto& consumer = old_tensor->consumer(0);
    for (size_t j = 0; j < consumer->num_inputs(); ++j) {
      if (consumer->input(j)->id() == old_tensor->id()) {
        Graph::ReplaceInput(consumer, j, new_tensor);
      }
    }
  }

  // 计算concat后param在buffer中的偏移
  // 并插入contiguous算子来实现到concat buffer的转移
  // 此时并不实际分配buffer
  if (_use_concat_buffer) {
    HT_ASSERT(_concat_buffer == nullptr)
      << "_concat_buffer shouldn't be initialized yet";
    // 2024.1.20 我们对本地已经拥有的param也要去做buffer
    TensorList buffer_comm_results;
    for (auto& comm_result : _comm_results) {
      HT_ASSERT(is_concat_op(comm_result->producer()))
        << "comm result should be concat op only";
      // ****TODO: A more general way!
      // now only support buffer when concat happens on a single axis while the other axis is not
      HandleConcatBuffer(comm_result, buffer_comm_results);
    }
    _concat_buffer = std::make_shared<ParamBuffer>("concat", buffer_comm_results);
    // HT_LOG_INFO << local_device << ": concat buffer has " << _concat_buffer->_tensor_list.size() << " tensor";
  }

  Graph::pop_graph_ctx();
  HT_LOG_DEBUG << local_device << ": make a new comm graph end...";
}

void SwitchExecGraph::BufferBatchedIsendIrecvExec(const hetu::impl::comm::NCCLCommunicationGroup& comm_nccl_group) {
  auto local_device = hetu::impl::comm::GetLocalDevice();
  NDArrayList send_data_list;
  NDArrayList recv_data_list;
  std::vector<hetu::impl::comm::CommTask> tasks;
  auto send_buffers_len = _send_buffers.size();
  auto recv_buffers_len = _recv_buffers.size();
  send_data_list.reserve(send_buffers_len);
  recv_data_list.reserve(recv_buffers_len);
  tasks.reserve(send_buffers_len + recv_buffers_len);
  for (const auto& kv : _send_buffers) {
    const auto& send_to_device = kv.first;
    auto send_data = kv.second->AsNDArray();
    HT_LOG_TRACE << local_device << " send to " << send_to_device << ": " << kv.second->tensor_list();
    tasks.push_back(comm_nccl_group->ISend(send_data, hetu::impl::comm::DeviceToWorldRank(send_to_device)));
    send_data_list.push_back(std::move(send_data));
  }
  for (const auto& kv : _recv_buffers) {
    const auto& recv_from_device = kv.first;
    auto recv_data = kv.second->AsNDArray();
    HT_LOG_TRACE << local_device << " recv from " << recv_from_device << ": " << kv.second->tensor_list();
    tasks.push_back(comm_nccl_group->IRecv(recv_data, hetu::impl::comm::DeviceToWorldRank(recv_from_device)));
    recv_data_list.push_back(std::move(recv_data));
  }
  comm_nccl_group->BatchedISendIRecv(tasks);
  NDArray::MarkUsedBy(send_data_list, Stream(local_device, kSwitchCollectiveStream));
  NDArray::MarkUsedBy(recv_data_list, Stream(local_device, kSwitchCollectiveStream));
  HT_LOG_DEBUG << local_device << ": BufferBatchedIsendIrecvExec is done";
}

void SwitchExecGraph::BufferBatchedIsendIrecv(const Operator& op,
                                              const hetu::impl::comm::NCCLCommunicationGroup& comm_nccl_group,
                                              Tensor2NDArrayMap& tensor2data,
                                              Tensor2IntMap& tensor2degrees) {
  auto local_device = hetu::impl::comm::GetLocalDevice();
  BatchedISendIRecvOpImpl& op_interface = dynamic_cast<BatchedISendIRecvOpImpl&>(op->body());
  auto op_stream = op->stream();
  // dup code
  /*
  const auto& comm_deivces = op_interface.comm_devices();
  std::vector<int> ranks(comm_deivces.size());
  std::transform(comm_deivces.begin(), comm_deivces.end(), ranks.begin(), 
                 [&](const Device& device) { return hetu::impl::comm::DeviceToWorldRank(device); });
  std::sort(ranks.begin(), ranks.end());
  auto& comm_group = hetu::impl::comm::NCCLCommunicationGroup::GetOrCreate(ranks, op_stream);
  */
  // 将原先input的NDArray移动并连续存储到各个send buffer中
  auto input_len = op->num_inputs();
  HT_ASSERT(input_len == op_interface.dst_devices().size())
    << "something wrong with the BatchedIsendIrecvOp input len";
  for (size_t i = 0; i < input_len; ++i) {
    const auto& input = op->input(i);
    auto it = tensor2data.find(input->id());
    HT_ASSERT(it != tensor2data.end() && it->second.is_defined())
      << local_device << ": Failed to execute the \"" << op->type() << "\" operation "
      << "(with name \"" << op->name() << "\"): "
      << "Cannot find input " << input;
    auto& data = it->second;
    HT_ASSERT(data->device() == input->placement() && data->dtype() == input->dtype())
      << local_device << ": Failed to execute the \"" << op->type() << "\" operation "
      << "(with name \"" << op->name() << "\"): "
      << "input " << input << " placement/dtype is wrong";
    HT_ASSERT(_send_buffers.find(op_interface.dst_devices()[i]) != _send_buffers.end())
      << "BufferBatchedIsendIrecv has no send buffer prepared for " << op_interface.dst_devices()[i];
    auto& buffer = _send_buffers[op_interface.dst_devices()[i]];
    // 那些bind的send buffer不需要管
    if (buffer->IsAuxiliary()) {
      if ((--tensor2degrees[input->id()]) == 0) {
        tensor2data.erase(input->id());
      }
      continue;
    }
    /*
    const size_t data_size = data->numel() * DataType2Size(data->dtype());
    auto buffer_data_storage = std::make_shared<NDArrayStorage>({buffer->AsRawPtr() + buffer_data_offset, 
                                                                data_size, local_device, 
                                                                static_cast<uint64_t>(-1)}); // set id to maximum
    */
    HT_LOG_TRACE << local_device << ": obtain buffer data for " << _info_mapping[input->id()]
      << ", whose shape is " << input->shape() << " and dtype is " << input->dtype() << " and tensor offset in buffer is " << buffer->GetElementOffset(input)
      << ", the whole size is " << buffer->AsStorage()->size();
    // 将原data移动到buffer中并转化成连续存储
    NDArrayMeta input_meta = input->meta();
    NDArrayMeta buffer_data_meta = input_meta.set_shape(input->shape()); // contiguous meta!!!
    auto buffer_data = NDArray(buffer_data_meta, buffer->AsStorage(), buffer->GetElementOffset(input));
    auto event = std::make_unique<hetu::impl::CUDAEvent>(local_device);
    auto stream = Stream(local_device, kSwitchComputingStream);
    event->Record(stream);
    _buffer_transfer_events.emplace_back(std::move(event));
    NDArray::contiguous(data, kSwitchComputingStream, buffer_data); // data ---> buffer_data
    NDArray::MarkUsedBy(data, stream);
    NDArray::MarkUsedBy(buffer_data, stream);
    event = std::make_unique<hetu::impl::CUDAEvent>(local_device);
    event->Record(stream);
    _buffer_transfer_events.emplace_back(std::move(event));
    // free memory after op async compute complete
    if ((--tensor2degrees[input->id()]) == 0) {
      tensor2data.erase(input->id());
    }
  }
  // 同步
  // 保证op_stream在kSwitchComputingStream之后
  auto buffer_transfer_events_len = _buffer_transfer_events.size() / 2;
  for (size_t i = 0; i < buffer_transfer_events_len; ++i) {
    // 用end event进行同步
    _buffer_transfer_events[2 * i + 1]->Block(op_stream);
  }
  // 分配recv的buffer
  // 等同步之后再分配，虽然无法overlap，会有latency上的损失，但是这样可以节省显存
  for (auto& kv : _recv_buffers) {
    auto& recv_buffer = kv.second;
    HT_ASSERT(!recv_buffer->IsAllocated())
      << "recv buffer shouldn't be allocated yet";
    // use_nccl=true will slow down
    recv_buffer->Alloc(op_stream, false, comm_nccl_group->GetComm());
  }
  op->instantiation_ctx().start[0]->Record(op_stream);
  BufferBatchedIsendIrecvExec(comm_nccl_group); // 执行BufferBatchedIsendIrecv
  op->instantiation_ctx().stop[0]->Record(op_stream);
  // HT_LOG_INFO << "BufferBatchedIsendIrecvExec end";
  // 清空send的buffer
  for (auto& kv : _send_buffers) {
    auto& send_buffer = kv.second;
    HT_ASSERT(send_buffer->IsAllocated())
      << "send buffer should be allocated";
    send_buffer->Free();
  }
  // 从各个连续的recv buffer取出离散的output的NDArray
  NDArrayList output_vals;
  auto output_len = op->num_outputs();
  HT_ASSERT(output_len == op_interface.src_devices().size())
    << "something wrong with the BatchedIsendIrecvOp output len";
  output_vals.reserve(output_len);
  for (size_t i = 0; i < output_len; ++i) {
    const auto& output = op->output(i);
    HT_ASSERT(_recv_buffers.find(op_interface.src_devices()[i]) != _recv_buffers.end())
      << "BufferBatchedIsendIrecv has no recv buffer prepared for " << op_interface.src_devices()[i];
    auto& buffer = _recv_buffers[op_interface.src_devices()[i]];
    auto buffer_data = NDArray(output->meta(), buffer->AsStorage(), buffer->GetElementOffset(output));
    tensor2data[output->id()] = buffer_data;
  }
}

// context switch
// 将before graph中的所有params以尽量高效的方式
// 重新分配到after graph中
void SwitchExecGraph::SwitchParams(SWITCH_MODE switch_mode, SWITCH_LEVEL switch_level) {

  // utils
  auto local_device = hetu::impl::comm::GetLocalDevice();
  auto is_feed_dict_op = [&](const Operator& op) -> bool {
    return Operator::all_output_tensors_of(op, [&](const Tensor& tensor) {
      return _comm_feed_dict.find(tensor->id()) != _comm_feed_dict.end();
    });
  };
  // 获取切换前后param的buffer
  std::shared_ptr<ParamBuffer> before_param_buffer;
  std::shared_ptr<ParamBuffer> after_param_buffer;
  if (switch_mode == SWITCH_MODE::SWITCH_ORIGIN_PARAM) {
    before_param_buffer = _switch_graph_pair.first->_origin_param_buffer;
    after_param_buffer = _switch_graph_pair.second->_origin_param_buffer;
  } else if (switch_mode == SWITCH_MODE::SWITCH_TRANSFER_PARAM) {
    before_param_buffer = _switch_graph_pair.first->_transfer_param_buffer;
    after_param_buffer = _switch_graph_pair.second->_transfer_param_buffer;
  } else if (switch_mode == SWITCH_MODE::SWITCH_CURRENT_GRAD) {
    before_param_buffer = _switch_graph_pair.first->_current_grad_buffer;
    after_param_buffer = _switch_graph_pair.second->_current_grad_buffer;
  } else if (switch_mode == SWITCH_MODE::SWITCH_ACCUMULATE_GRAD) {
    before_param_buffer = _switch_graph_pair.first->_accumulate_grad_buffer;
    after_param_buffer = _switch_graph_pair.second->_accumulate_grad_buffer;
  } else {
    HT_RUNTIME_ERROR << "NotImplementedError";
  }
  if (switch_level != SWITCH_LEVEL::TOPO) {
    HT_ASSERT(before_param_buffer->IsAllocated() && !after_param_buffer->IsAllocated())
      << "wrong allocation state for buffers"
      << ", before param buffer alloc state is " << before_param_buffer->IsAllocated()
      << " and after param buffer alloc state is " << after_param_buffer->IsAllocated();
    HT_ASSERT(!before_param_buffer->IsEmpty() && !after_param_buffer->IsEmpty())
      << "wrong empty state for buffers";
    // 为grad分配_preserved_data
    // 因为要对齐热切换的接口
    if (switch_mode == SWITCH_MODE::SWITCH_CURRENT_GRAD
        || switch_mode == SWITCH_MODE::SWITCH_ACCUMULATE_GRAD) {
      for (const auto& before_param : before_param_buffer->tensor_list()) {
        auto before_param_data = NDArray(before_param->meta(),
                                         before_param_buffer->AsStorage(), 
                                         before_param_buffer->GetElementOffset(before_param));
        _switch_graph_pair.first->_preserved_data[before_param->id()] = before_param_data;
      }
    }
  }

  // 如果有cache好的_comm_graph
  // 那么直接使用即可
  // 否则需要重新建立
  if (_comm_graph != nullptr) {
    // 如果只需要建立topo
    // 此处直接返回即可
    if (switch_level == SWITCH_LEVEL::TOPO) {
      return;
    }
    // 只需要重新设置_comm_feed_dict即可
    // 从_preserve_data中获取before graph的params的数据
    for (const auto& kv : _comm_feed_dict_mapping) {
      auto comm_input_data_it = _switch_graph_pair.first->_preserved_data.find(kv.second->id());
      HT_ASSERT(comm_input_data_it != _switch_graph_pair.first->_preserved_data.end())
        << "something wrong, the data to transfer in the before graph is not available";
      // 给feed_dict赋上NDArray
      _comm_feed_dict[kv.first] = {comm_input_data_it->second};
    }
  } else {
    TIK(switch_params_making);
    HT_ASSERT(_comm_results.empty())
      << "no comm result should exist";
    // *建图*
    MakeCommGraph(switch_mode, switch_level);
    // 计算topo
    HT_LOG_DEBUG << local_device << ": the mutual params len is " << _param_blocks.size()
      << " and the local recv params len is " << _comm_results.size();
    TensorList fetches(_comm_results);
    fetches.insert(fetches.end(), _dummy_links.begin(), _dummy_links.end());
    OpRefList topo = Graph::TopoSort(fetches, -1, is_feed_dict_op);
    // HT_LOG_DEBUG << local_device << ": global topo of the comm graph is " << topo;
    // 本地topo
    auto get_local_topo = [&](OpRefList& topo, OpRefList& local_topo) {
      for (auto& op_ref : topo) {
        HT_ASSERT(op_ref.get()->placement().type() != DeviceType::UNDETERMINED)
          << "op " << op_ref.get() << " in comm graph is not instantiated";
        if (op_ref.get()->placement() == local_device) {
          local_topo.push_back(op_ref);
        }
      }
    };
    get_local_topo(topo, _comm_topo);
    HT_LOG_DEBUG << local_device << ": local topo of the comm graph is " << _comm_topo;
    // 计算运行时shape
    // 该图中只存在placeholder、split、batchedisendirecv和concatenate
    // 不需要symbolic方法（甚至不需要DoInferShape）
    // 直接用tensor的shape即可
    // Question: 是否正确？
    for (auto& op_ref : _comm_topo) {
      auto& op = op_ref.get();
      for (const auto& output : op->outputs()) {
        _comm_shape_plan[output->id()] = output->shape();
      }
    }
    // *TODO: 调整topo使得BatchedIsendIrecv靠前
    // 否则有可能concat buffer被先alloc而send buffer还未被释放（会增加显存占用）
    // 结束对准备工作的profile
    if (_profile_level < SWITCH_PROFILE_LEVEL::INFO) {
      TOK(switch_params_making);
      HT_LOG_INFO << local_device << ": switch params making graph & plan time = " << COST_MSEC(switch_params_making) << " ms";
    }
  }

  if (_profile_level <= SWITCH_PROFILE_LEVEL::MEMORY) {
    ProfileMemory("switch exec graph before warm up");
  }
  // 热启动nccl的all-to-all通信
  // warm up the comm group
  DeviceGroup comm_device_group{std::vector<Device>(_comm_set.begin(), _comm_set.end())};
  std::vector<int> comm_ranks(_comm_set.size());
  std::transform(_comm_set.begin(), _comm_set.end(), comm_ranks.begin(), 
                 [&](const Device& device) { return hetu::impl::comm::DeviceToWorldRank(device); });
  std::sort(comm_ranks.begin(), comm_ranks.end());
  auto& comm_nccl_group = hetu::impl::comm::NCCLCommunicationGroup::GetOrCreate(comm_ranks, Stream(local_device, kSwitchCollectiveStream));
  std::call_once(warmup_flags[comm_device_group], 
                 WarmUpComm, 
                 _comm_set);
  // 如果只需要建立topo
  // 此处直接返回即可
  if (switch_level == SWITCH_LEVEL::TOPO) {
    return;
  }

  // profile时需要先都同步好了
  if (_profile_level < SWITCH_PROFILE_LEVEL::INFO) {
    // stream同步
    SynchronizeAllStreams(local_device);
    // rank同步
    // 这里对全部rank进行同步
    /*
    if (!_comm_set.empty()) {
      std::vector<Device> mpi_devices(_comm_set.begin(), _comm_set.end());
      DeviceGroup mpi_device_group{mpi_devices};
      auto& mpi_group = hetu::impl::comm::MPICommunicationGroup::GetOrCreate(hetu::impl::comm::DeviceGroupToWorldRanks(mpi_device_group));
      mpi_group->Barrier(true);
      HT_LOG_DEBUG << local_device << ": params switch comm set = " << mpi_device_group;
    }
    */
    auto& mpi_group = hetu::impl::comm::MPICommunicationGroup::GetOrCreateWorldwide();
    mpi_group->Barrier(true);
    if (_profile_level <= SWITCH_PROFILE_LEVEL::MEMORY) {
      ProfileMemory("switch exec graph begin");
    }
    if (_profile_level <= SWITCH_PROFILE_LEVEL::NVLINK) {
      ProfileNvlinkStart();
    }
    mpi_group->Barrier(true);
  }
  
  // 启动！
  TIK(switch_params_running); // 开始计时
  Tensor2NDArrayMap tensor2data;
  Tensor2IntMap tensor2degrees;
  std::set<TensorId> useful_tensors;
  RuntimeContext runtime_ctx(_comm_topo.size(), _comm_shape_plan);
  // TODO: 这一部分也可以和topo一样cache下来
  // 计算各个tensor的度以及topo中涉及的tensor
  for (auto& op_ref : _comm_topo) {
    for (auto& input : op_ref.get()->inputs()) {
      tensor2degrees[input->id()]++;
    }
    for (auto& output : op_ref.get()->outputs()) {
      useful_tensors.insert(output->id());
    }
  }
  // 释放没有用的_comm_feed_dict和_preserved_data
  auto feed_dict_it = _comm_feed_dict.begin();
  while (feed_dict_it != _comm_feed_dict.end()) {
    auto tensor_id = feed_dict_it->first;
    if (useful_tensors.find(tensor_id) == useful_tensors.end()) {
      // 清feed_dict
      feed_dict_it = _comm_feed_dict.erase(feed_dict_it);
      // 清before_graph的_preserved_data
      _switch_graph_pair.first->_preserved_data.erase(_comm_feed_dict_mapping[tensor_id]->id());
      continue;
    }
    feed_dict_it++;
  }
  // 释放before param buffer
  // 此时并不会真正free，因为还有_preserved_data
  // 只是交出所有权
  before_param_buffer->Free(); 
  // 分配send的buffer
  for (auto& kv : _send_buffers) {
    auto& send_to_device = kv.first;
    auto& send_buffer = kv.second;
    HT_ASSERT(!send_buffer->IsAllocated()) 
      << "send buffer shouldn't be allocated yet";
    // 一个优化
    // 如果向两个device发送的buffer要send的slice一模一样，那么只存一份即可
    // 例如最极端情况下tp=8->dp=8，7条send buffer实际上是完全一样的
    // TODO: 后续可以考虑设计算法找公共子序列作为实际的buffer
    for (auto& old_kv : _send_buffers) {
      auto& old_send_to_device = old_kv.first;
      auto& old_send_buffer = old_kv.second;
      if (old_send_to_device != send_to_device
          && old_send_buffer->IsAllocated()
          && old_send_buffer->IsEqual(*send_buffer)) {
        send_buffer->Bind(old_send_buffer->AsStorage());
        break;
      }
    }
    if (!send_buffer->IsAllocated()) {
      // use_nccl=true will slow down
      send_buffer->Alloc(Stream(local_device, kSwitchCollectiveStream), false, comm_nccl_group->GetComm());
    }
  }
  if (_profile_level <= SWITCH_PROFILE_LEVEL::MEMORY) {
    ProfileMemory("switch exec graph after alloc send buffers");
  }
  // 运行topo
  for (auto& op_ref : _comm_topo) {
    auto& op = op_ref.get();
    HT_LOG_DEBUG << local_device << ": handling op " << op << " in comm graph";
    // 特殊处理1
    // 对于feed_dict需要清除原先的data映射并将其放在tensor2data的intermediate的映射中
    if (is_feed_dict_op(op)) {
      // 可以保证这里全都是只有一个输出的placeholder
      HT_ASSERT(is_placeholder_op(op))
        << "feed dict op must be a placeholder in the comm graph";
      auto tensor_id = op->output(0)->id();
      tensor2data[tensor_id] = _comm_feed_dict[tensor_id][0];
      // 清feed_dict
      _comm_feed_dict.erase(tensor_id);
      // 清before_graph的_preserved_data
      _switch_graph_pair.first->_preserved_data.erase(_comm_feed_dict_mapping[tensor_id]->id());
      continue;
    }
    // 特殊处理2
    // 使用聚合的buffer进行通信
    if (is_batched_isend_irecv_op(op)) {
      BufferBatchedIsendIrecv(op, comm_nccl_group, tensor2data, tensor2degrees);
      continue;
    }
    // 特殊处理3
    // 分配after graph的param的runtime allocation（_concat_buffer)
    // 1)、>=2输入的concat算子的输出（目前我们的场景中一个after param只会对应一个这样的concat所以问题不大）
    // 2)、由于要转移recv buffer而不得不插入的contiguous算子
    if (_use_concat_buffer && (is_concat_op(op) || is_contiguous_op(op)) && _concat_buffer->HasTensor(op->output(0))) {
      HT_ASSERT(_concat_buffer != nullptr)
        << "_concat_buffer should be initialized";
      // Alloc on-the-fly
      if (!_concat_buffer->IsAllocated()) {
        // 分配concat的buffer
        // TODO: 目前可能显存溢出，之后应该考虑在溢出时扫描mempool中的free_event
        // 目前kSwitchComputingStream的concat_buffer的显存只能确保可以重用send_buffer释放出的显存
        _concat_buffer->Alloc(Stream(local_device, kSwitchComputingStream));
      }
      auto& output = op->output(0);
      auto output_data = NDArray(output->meta(), _concat_buffer->AsStorage(), _concat_buffer->GetElementOffset(output));
      runtime_ctx.add_runtime_allocation(output->id(), output_data);
    }
    // 正常按照算子的逻辑进行处理
    NDArrayList input_vals;
    input_vals.reserve(op->num_inputs());
    for (const auto& input : op->inputs()) {
      auto it = tensor2data.find(input->id());
      HT_ASSERT(it != tensor2data.end() && it->second.is_defined())
        << local_device << ": Failed to execute the \"" << op->type() << "\" operation "
        << "(with name \"" << op->name() << "\"): "
        << "Cannot find input " << input;
      auto& data = it->second;
      HT_ASSERT(data->device() == input->placement() && data->dtype() == input->dtype())
        << local_device << ": Failed to execute the \"" << op->type() << "\" operation "
        << "(with name \"" << op->name() << "\"): "
        << "input " << input << " placement/dtype is wrong";
      input_vals.push_back(data);
      // free memory after op async compute complete
      if ((--tensor2degrees[input->id()]) == 0) {
        tensor2data.erase(input->id());
      }
    }
    NDArrayList output_vals = op->Compute(input_vals, runtime_ctx);
    // Note: The usage should be marked inside kernels, 
    // but we still mark here in case we forget to do so in some kernels. 
    NDArray::MarkUsedBy(input_vals, op->instantiation_ctx().stream());
    NDArray::MarkUsedBy(output_vals, op->instantiation_ctx().stream());
    // 记录输出值
    for (size_t i = 0; i < op->num_outputs(); ++i) {
      tensor2data[op->output(i)->id()] = output_vals[i];
      // 如果是最后的输出
      // 我们记录一些async的event
      // 这样可以在下一个exec graph跑的时候再进行同步
      auto it = _comm_results_mapping.find(op->output(i)->id());
      if (it != _comm_results_mapping.end()) {
        // 如果是param
        if (switch_mode == SWITCH_MODE::SWITCH_ORIGIN_PARAM 
            || switch_mode == SWITCH_MODE::SWITCH_TRANSFER_PARAM) {
          auto event = std::make_unique<hetu::impl::CUDAEvent>(op->placement());
          event->Record(op->instantiation_ctx().stream());
          _switch_graph_pair.second->_switch_param_events[it->second->id()] = std::move(event);
        }
        // 如果是grad
        else {
          auto event = std::make_unique<hetu::impl::CUDAEvent>(op->placement());
          event->Record(op->instantiation_ctx().stream());
          _switch_graph_pair.second->_switch_grad_events[it->second->id()] = std::move(event);
        }
      }
    }
  }
  HT_LOG_DEBUG << "switch exec graph topo end";
  // 将结果赋值给after graph
  for (const auto& kv : _comm_results_mapping) {
    auto it = tensor2data.find(kv.first);
    HT_ASSERT(it != tensor2data.end())
      << "something wrong, can't find the result from the tensor2data mapping";
    HT_LOG_DEBUG << local_device << ": comm result sum of " << kv.second << " is " << NDArray::sum(it->second);
    // 给新图的_preserved_data赋上NDArray
    // 对于grad则不需要（_preserved_data表示已经算出来的）
    if (switch_mode == SWITCH_MODE::SWITCH_ORIGIN_PARAM 
        || switch_mode == SWITCH_MODE::SWITCH_TRANSFER_PARAM) {
      _switch_graph_pair.second->_preserved_data[kv.second->id()] = it->second;
    }
    // allocation check
    if (_use_concat_buffer) {
      // 值得一提的是after param并不在concat buffer的tensor list中
      // 因为tensor list存的是comm graph中的一些中间tensor
      // 此处只能验证storage是共享的
      HT_ASSERT(_concat_buffer->AsStorage() == it->second->storage())
        << local_device << ": after param " << kv.second << " is not allocated in the concat buffer";
    }
  }
  HT_ASSERT(_comm_feed_dict.empty())
    << "comm feed dict should be empty now"
    << ", but it turns out to be " << _comm_feed_dict;
  for (auto& kv : _switch_graph_pair.first->_preserved_data) {
    HT_ASSERT(!before_param_buffer->HasTensor(kv.first))
      << "before graph preserved data is not cleaned up, still remains " << before_param_buffer->GetTensor(kv.first)
      << ", which will cause the failure of releasing the before graph buffer";
  }
  // 清空tensor2data
  tensor2data.clear();
  // 清空recv的buffer
  // TODO: 按bucket发送并及时清除（虽然会提高延时，但能降低显存）
  for (auto& kv : _recv_buffers) {
    auto& recv_buffer = kv.second;
    HT_ASSERT(recv_buffer->IsAllocated())
      << "recv buffer should be allocated";
    recv_buffer->Free();
  }
  // 清空concat的buffer
  // 让_concat_buffer的storage的所有权转交给after param buffer和_preserved_data
  if (_use_concat_buffer) {
    // HT_LOG_INFO << local_device << ": after_param_buffer tensor list is " << after_param_buffer->_tensor_list;
    // HT_LOG_INFO << local_device << ": conat_buffer tensor list is " << _concat_buffer->_tensor_list;
    after_param_buffer->Bind(_concat_buffer->AsStorage());
    _concat_buffer->Free();
  }
  // 已在处理feed_dict时自动清除
  /*
  // 将before graph中保留的数据全部清除
  // TODO: 最坏情况需要1.5倍的显存开销，后续需要分bucket进行发送并清除
  _switch_graph_pair.first->_preserved_data.clear();
  */

  // 非profile情形下这里不需要同步
  // 下一个exec graph需要某一个param时再进行sync
  
  if (_profile_level < SWITCH_PROFILE_LEVEL::INFO) {
    // stream同步
    TensorList fetches(_comm_results);
    fetches.insert(fetches.end(), _dummy_links.begin(), _dummy_links.end());
    for (const auto& fetch : fetches) {
      fetch->producer()->Sync();
    }  
    TOK(switch_params_running); // 结束计时
    // rank同步（不同rank耗时不一样，因此放在TOK之后）
    // 这里对全部rank进行同步
    /*
    if (!_comm_set.empty()) {
      std::vector<Device> mpi_devices(_comm_set.begin(), _comm_set.end());
      DeviceGroup mpi_device_group{mpi_devices};
      auto& mpi_group = hetu::impl::comm::MPICommunicationGroup::GetOrCreate(hetu::impl::comm::DeviceGroupToWorldRanks(mpi_device_group));
      mpi_group->Barrier(true);
    }
    */
    auto& mpi_group = hetu::impl::comm::MPICommunicationGroup::GetOrCreateWorldwide();
    mpi_group->Barrier(true);
    std::string log_prefix = (switch_mode == SWITCH_MODE::SWITCH_ORIGIN_PARAM ||switch_mode == SWITCH_MODE::SWITCH_TRANSFER_PARAM) ?
                             "param" : "grad";
    HT_LOG_INFO << local_device << ": " << log_prefix << " switch running time = " << COST_MSEC(switch_params_running) << " ms";
    char* switch_log_file = std::getenv("HETU_SWITCH_LOG_FILE");
    if (switch_log_file != nullptr && hetu::impl::comm::GetWorldRank() == 0) {
      std::ofstream file;
      file.open(switch_log_file, std::ios_base::app);
      if (file.is_open()) {
        file << COST_MSEC(switch_params_running) << " ms";
        file.close();
      } else {
        HT_RUNTIME_ERROR << "Error opening the file";
      }
    }
    if (_profile_level <= SWITCH_PROFILE_LEVEL::MEMORY) {
      ProfileMemory("switch exec graph end");
    }
    if (_profile_level <= SWITCH_PROFILE_LEVEL::NVLINK) {
      ProfileNvlinkEnd();
    }
    ProfileRunningDetails();
  }
  
  std::string log_prefix = (switch_mode == SWITCH_MODE::SWITCH_ORIGIN_PARAM ||switch_mode == SWITCH_MODE::SWITCH_TRANSFER_PARAM) ?
                           "param" : "grad";
  HT_LOG_INFO << local_device << ": " << log_prefix << " switch from " << _switch_graph_pair.first->name()
   << " to " << _switch_graph_pair.second->name() << " is done (async)";
  // HT_RUNTIME_ERROR << local_device << ": breakpoint";
}

// profile details
void SwitchExecGraph::ProfileRunningDetails() {

  HT_ASSERT(_comm_graph != nullptr)
    << "Profiler can only used after comm graph was built";
  auto local_device = hetu::impl::comm::GetLocalDevice();
  size_t slice_num = 0, concat_num = 0, comm_num = 0;
  double slice_time = 0, concat_time = 0, comm_time = 0;
  size_t send_buffer_transfer_num = 0, concat_buffer_transfer_num = 0;
  double send_buffer_transfer_time = 0, concat_buffer_transfer_time = 0;
  size_t comm_buffer_num = 0;
  size_t comm_buffer_alloc_time = 0, comm_buffer_free_time = 0;
  size_t concat_buffer_num = 0;
  size_t concat_buffer_alloc_time = 0, concat_buffer_free_time = 0;
  
  // op execute time
  for (auto& op_ref : _comm_topo) {
    auto& op = op_ref.get();
    // Note: only one micro batch
    if (is_placeholder_op(op)) {
      continue;
    } else if (is_slice_op(op)) {
      slice_time += op->TimeCost(0) * 1.0 / 1e6;
      slice_num += 1;
    } else if (is_concat_op(op)) {
      // inplace and no concat
      if (op->num_inputs() == 1) {
        continue;
      }
      concat_time += op->TimeCost(0) * 1.0 / 1e6;
      concat_num += 1;
    } else if (is_contiguous_op(op)) {
      concat_buffer_transfer_time += op->TimeCost(0) * 1.0 / 1e6;
      concat_buffer_transfer_num += 1;
    } else if (is_batched_isend_irecv_op(op)) {
      comm_time += op->TimeCost(0) * 1.0 / 1e6;
      comm_num += 1;
    } else {
      HT_RUNTIME_ERROR << local_device << ": op " << op 
        << " shouldn't exist in the comm graph";
    }
  }

  // buffer transfer time
  send_buffer_transfer_num = _buffer_transfer_events.size() / 2;
  for (size_t i = 0; i < send_buffer_transfer_num / 2; ++i) {
    auto& start = _buffer_transfer_events[2 * i];
    auto& end = _buffer_transfer_events[2 * i + 1];
    // record the time of buffer transfer
    send_buffer_transfer_time += end->TimeSince(*start) * 1.0 / 1e6;
  }

  // buffer alloc/free time
  for (const auto& kv : _send_buffers) {
    comm_buffer_num += 1;
    comm_buffer_alloc_time += kv.second->_alloc_time;
    comm_buffer_free_time += kv.second->_free_time;
  }
  for (const auto& kv : _recv_buffers) {
    comm_buffer_num += 1;
    comm_buffer_alloc_time += kv.second->_alloc_time;
    comm_buffer_free_time += kv.second->_free_time;
  }
  if (_use_concat_buffer) {
    concat_buffer_num += 1;
    concat_buffer_alloc_time += _concat_buffer->_alloc_time;
    concat_buffer_free_time += _concat_buffer->_free_time;
  }

  // comm detailed info
  std::vector<Device>& send_to_devices = _send_mapping[local_device].first;
  std::vector<Device>& recv_from_devices = _recv_mapping[local_device].first;
  TensorList& send_tensors = _send_mapping[local_device].second;
  TensorList& recv_tensors = _recv_mapping[local_device].second;
  std::unordered_map<Device, std::vector<std::string>> send_info_mapping;
  std::unordered_map<Device, std::vector<std::string>> recv_info_mapping;
  std::ostringstream send_info_output;
  std::ostringstream recv_info_output;
  auto send_len = send_to_devices.size();
  auto recv_len = recv_from_devices.size();
  HT_ASSERT(send_tensors.size() == send_len)
    << "something wrong with the send size";
  HT_ASSERT(recv_tensors.size() == recv_len)
    << "something wrong with the recv size";
  for (size_t i = 0; i < send_len; ++i) {
    auto it = _info_mapping.find(send_tensors[i]->id());
    HT_ASSERT(it != _info_mapping.end())
      << "send tensor info is not existed";
    send_info_mapping[send_to_devices[i]].push_back(it->second);
  }
  for (size_t i = 0; i < recv_len; ++i) {
    auto it = _info_mapping.find(recv_tensors[i]->id());
    HT_ASSERT(it != _info_mapping.end())
      << "recv tensor info is not existed";
    recv_info_mapping[recv_from_devices[i]].push_back(it->second);
  }
  for (const auto& kv : send_info_mapping) {
    send_info_output << "send " << kv.second.size()
      << " tensor to " << kv.first;
    if (_profile_level == SWITCH_PROFILE_LEVEL::TRACE) {
      for (const auto& send_info : kv.second) {
        send_info_output << ", " << send_info;
      }
    }
    send_info_output << std::endl;
  }
  for (const auto& kv : recv_info_mapping) {
    recv_info_output << "recv " << kv.second.size()
      << " tensor from " << kv.first;
    if (_profile_level == SWITCH_PROFILE_LEVEL::TRACE) {
      for (const auto& recv_info : kv.second) {
        recv_info_output << ", " << recv_info;
      }
    }
    recv_info_output << std::endl;
  }

  HT_LOG_INFO << local_device << ": switch params running details: " << std::endl
    << "*********************************************" << std::endl
    << "comm buffer num = " << comm_buffer_num << ", alloc time = " << comm_buffer_alloc_time << " ms" << std::endl
    << "comm buffer num = " << comm_buffer_num << ", free time = " << comm_buffer_free_time << " ms" << std::endl
    << "concat buffer num = " << concat_buffer_num << ", alloc time = " << concat_buffer_alloc_time << " ms" << std::endl
    << "concat buffer num = " << concat_buffer_num << ", free time = " << concat_buffer_free_time << " ms" << std::endl
    << "*********************************************" << std::endl
    << "slice num = " << slice_num << ", time = " << slice_time << " ms" << std::endl
    << "concat num = " << concat_num << ", time = " << concat_time << " ms" << std::endl
    << "send buffer transfer num = " << send_buffer_transfer_num << ", time = " << send_buffer_transfer_time << " ms" << std::endl
    << "concat buffer transfer num = " << concat_buffer_transfer_num << ", time = " << concat_buffer_transfer_time << " ms" << std::endl
    << "comm num = " << comm_num << ", time = " << comm_time << " ms" << std::endl
    << "*********************************************" << std::endl
    << send_info_output.str()
    << "---------------------------------------------" << std::endl
    << recv_info_output.str()
    << "*********************************************";
}

void SwitchExecGraph::ProfileNvlinkStart() {

  // 只需要一个机器profile即可
  if (hetu::impl::comm::GetWorldRank() != 0) {
    return;
  }
  HT_LOG_INFO << "********* Profile NVLink Start *********";

  // 初始化NVML库
  nvmlReturn_t result = nvmlInit();
  if (result != NVML_SUCCESS) {
    HT_RUNTIME_ERROR << "Failed to initialize NVML: " << nvmlErrorString(result);
    return;
  }

  // 获取GPU数量
  result = nvmlDeviceGetCount(&_device_count);
  if (result != NVML_SUCCESS) {
    HT_RUNTIME_ERROR << "Failed to query device count: " << nvmlErrorString(result);
    return;
  }
  _nvlink_counts.reserve(_device_count);
  _nvlink_txs.reserve(_device_count);
  _nvlink_rxs.reserve(_device_count);

  for (unsigned int i = 0; i < _device_count; ++i) {
    // Initialization
    _nvlink_counts.emplace_back(0);
    _nvlink_txs.emplace_back();
    _nvlink_rxs.emplace_back();

    // Get current device
    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(i, &device);
    if (result != NVML_SUCCESS) {
      HT_RUNTIME_ERROR << "Failed to get handle for device " << i << ": " << nvmlErrorString(result);
      return;
    }

    // Check the NVLink status for each possible link
    for (unsigned int link = 0; link < NVML_NVLINK_MAX_LINKS; ++link) {
      nvmlEnableState_t is_active;
      result = nvmlDeviceGetNvLinkState(device, link, &is_active);
      if (NVML_SUCCESS == result && is_active == NVML_FEATURE_ENABLED) {
        _nvlink_counts[i]++;
      }
    }
    HT_LOG_INFO << "GPU " << i << " has " << _nvlink_counts[i] << " NVLink connections active";
    if (_nvlink_counts[i] == 0) {
      continue;
    }
    _nvlink_txs.reserve(_nvlink_counts[i]);
    _nvlink_rxs.reserve(_nvlink_counts[i]);

    // 创建NVML字段值数组
    std::vector<nvmlFieldValue_t> field_values(2 * _nvlink_counts[i]);
    for (unsigned int link = 0; link < _nvlink_counts[i]; ++link) {
      field_values[2 * link].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX;
      field_values[2 * link].scopeId = link; // 设置scopeId为linkId
      field_values[2 * link + 1].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX;
      field_values[2 * link + 1].scopeId = link; // 设置scopeId为linkId
    }

    // 记录执行nccl通信代码片段前的NVLink Raw Tx和Raw Rx
    result = nvmlDeviceGetFieldValues(device, field_values.size(), field_values.data());
    if (result != NVML_SUCCESS) {
      HT_RUNTIME_ERROR << "Failed to get utilization control: " << nvmlErrorString(result);
      return;
    }
    for (unsigned int link = 0; link < _nvlink_counts[i]; ++link) {
      _nvlink_txs[i].emplace_back(field_values[2 * link].value.ullVal);
      _nvlink_rxs[i].emplace_back(field_values[2 * link + 1].value.ullVal);
    }
  }
}


void SwitchExecGraph::ProfileNvlinkEnd() {

  // 只需要一个机器profile即可
  // 如果没有NVLink则不再profile
  if (hetu::impl::comm::GetWorldRank() != 0) {
    return;
  }

  nvmlReturn_t result;
  for (unsigned int i = 0; i < _device_count; ++i) {
    if (_nvlink_counts[i] == 0) {
      continue;
    }
    // Get current device
    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(i, &device);
    if (result != NVML_SUCCESS) {
      HT_RUNTIME_ERROR << "Failed to get handle for device " << i << ": " << nvmlErrorString(result);
      return;
    }

    // 创建NVML字段值数组
    std::vector<nvmlFieldValue_t> field_values(2 * _nvlink_counts[i]);
    for (unsigned int link = 0; link < _nvlink_counts[i]; ++link) {
      field_values[2 * link].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX;
      field_values[2 * link].scopeId = link; // 设置scopeId为linkId
      field_values[2 * link + 1].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX;
      field_values[2 * link + 1].scopeId = link; // 设置scopeId为linkId
    }

    // 获取执行nccl通信代码片段后的NVLink Raw Tx和Raw Rx
    result = nvmlDeviceGetFieldValues(device, field_values.size(), field_values.data());
    if (result != NVML_SUCCESS) {
      HT_RUNTIME_ERROR << "Failed to get utilization control: " << nvmlErrorString(result);
      return;
    }
    for (unsigned int link = 0; link < _nvlink_counts[i]; ++link) {
      _nvlink_txs[i][link] = field_values[2 * link].value.ullVal - _nvlink_txs[i][link];
      _nvlink_rxs[i][link] = field_values[2 * link + 1].value.ullVal - _nvlink_rxs[i][link];
      // 打印NVLink Raw Tx和Raw Rx的变化量
      HT_LOG_INFO << "GPU " << i << " NVLink " << link << " Data Tx Delta: " << _nvlink_txs[i][link] << " KiB";
      HT_LOG_INFO << "GPU " << i << " NVLink " << link << " Data Rx Delta: " << _nvlink_rxs[i][link] << " KiB";
    }
  }

  // 清理NVML资源
  result = nvmlShutdown();
  if (result != NVML_SUCCESS) {
    HT_RUNTIME_ERROR << "Failed to shutdown NVML: " << nvmlErrorString(result);
    return;
  }
   
  HT_LOG_INFO << "********* Profile NVLink End *********";
}

void SwitchExecGraph::ProfileMemory(const std::string& prefix) {
  // 初始化NVML库
  nvmlReturn_t result = nvmlInit();
  if (result != NVML_SUCCESS) {
    HT_RUNTIME_ERROR << "Failed to initialize NVML: " << nvmlErrorString(result);
    return;
  }

  nvmlDevice_t device;
  auto device_index = hetu::impl::comm::GetLocalDevice().index();
  result = nvmlDeviceGetHandleByIndex(device_index, &device);
  if (result != NVML_SUCCESS) {
    HT_RUNTIME_ERROR << "Failed to get device handle: " << nvmlErrorString(result);
    return;
  }

  nvmlMemory_t memory;
  result = nvmlDeviceGetMemoryInfo(device, &memory);
  if (result != NVML_SUCCESS) {
    HT_RUNTIME_ERROR << "Failed to get memory info: " << nvmlErrorString(result);
    return;
  }
        
  HT_LOG_DEBUG << "[" << prefix << "] Device " << hetu::impl::comm::GetWorldRank() << ": ";
  // HT_LOG_INFO << "Total Memory: " << memory.total / (1024 * 1024) << " MiB";
  HT_LOG_DEBUG << "Used Memory: " << memory.used / (1024 * 1024) << " MiB";
  // HT_LOG_INFO << "Free Memory: " << memory.free / (1024 * 1024) << " MiB";

  result = nvmlShutdown();
  if (result != NVML_SUCCESS) {
    HT_RUNTIME_ERROR << "Failed to shutdown NVML: " << nvmlErrorString(result);
    return;
  }
}

} // namespace graph
} // namespace hetu
