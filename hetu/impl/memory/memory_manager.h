#pragma once

#include <memory>
#include <cstddef>

namespace hetu {
namespace impl {

struct MemoryBlock {
  void* ptr;
  size_t size;
  bool allocated;
  std::shared_ptr<MemoryBlock> prev;
  std::shared_ptr<MemoryBlock> next;

  MemoryBlock(void* ptr_, size_t size_, bool allocated_ = false)
    : ptr(ptr_), size(size_), allocated(allocated_), prev(nullptr), next(nullptr) {}
};

class MemoryManager {
 public:
  MemoryManager(void* begin_ptr, size_t total_size)
    : _begin_ptr(begin_ptr), _total_size(total_size) {
    // Initialize the memory manager with a single large free block
    _head_block = std::make_shared<MemoryBlock>(begin_ptr, total_size);
  }

  ~MemoryManager() {
    // No need for explicit cleanup, shared_ptr will handle it
  }

  bool Malloc(void** ptr, size_t size);
  void Free(void* ptr);

 protected:
  void* _begin_ptr;
  size_t _total_size;
  std::shared_ptr<MemoryBlock> _head_block;
};

} // namespace impl
} // namespace hetu