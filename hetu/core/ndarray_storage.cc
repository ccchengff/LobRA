#include "hetu/core/ndarray_storage.h"
#include "hetu/impl/utils/cuda_utils.h"
#include <nccl.h>

namespace hetu {

static size_t wrap_num = 0;
static std::shared_ptr<ncclGroupMemCtx> global_nccl_group_mem_ctx = nullptr;

void ncclGroupStart_safe() {
  wrap_num++;
  NCCL_CALL(ncclGroupStart());
  if (wrap_num >= 2) {
    HT_ASSERT(global_nccl_group_mem_ctx)
      << "global_nccl_group_mem_ctx should already existed when wrapping ncclGroupStart(End), just return then";
    return;
  }
  global_nccl_group_mem_ctx = std::make_shared<ncclGroupMemCtx>();
}

void ncclGroupEnd_safe() {
  HT_ASSERT(wrap_num >= 1)
    << "ensure you have called ncclGroupStart_safe() in advance";
  wrap_num--;
  NCCL_CALL(ncclGroupEnd());
  HT_ASSERT(global_nccl_group_mem_ctx)
    << "Must have global_nccl_group_mem_ctx, ensure you have called ncclGroupStart_safe() in advance";
  global_nccl_group_mem_ctx->Free();
  if (wrap_num == 0) {
    global_nccl_group_mem_ctx = nullptr;
  }
}

NDArrayStorage::~NDArrayStorage() {
  if (_in_mempool) {
    // 如果在ncclGroupStart与End中需要滞后释放
    if (global_nccl_group_mem_ctx) {
      global_nccl_group_mem_ctx->AddFreeDataPtr(_ptr);
    } 
    // 立即释放
    else {
      FreeToMemoryPool(_ptr);
    }
  } 
  // deprecated: 这一部分目前可以使用mempool的borrow data
  else {
    // 内存由外界维护
    // 例如ncclMemAlloc和ncclMemFree
    return;
  }
}

} // namespace hetu