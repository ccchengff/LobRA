#include "hetu/impl/memory/memory_manager.h"
#include "hetu/common/except.h"

namespace hetu {
namespace impl {

bool MemoryManager::Malloc(void** ptr, size_t size) {
  auto curr_block = _head_block;
  while (curr_block) {
    // Found a suitable block
    if (!curr_block->allocated && curr_block->size >= size) {
      // Split the block if needed
      if (curr_block->size > size) {
        auto new_block = std::make_shared<MemoryBlock>(
          static_cast<char*>(curr_block->ptr) + size,
          curr_block->size - size,
          false
        );
        new_block->prev = curr_block;
        new_block->next = curr_block->next;
        if (curr_block->next) {
          curr_block->next->prev = new_block;
        }
        curr_block->next = new_block;
        curr_block->size = size;
      }
      curr_block->allocated = true;
      *ptr = curr_block->ptr;
      return true;
    }
    curr_block = curr_block->next;
  }
  return false; // No suitable block found
}

void MemoryManager::Free(void* ptr) {
  auto curr_block = _head_block;
  while (curr_block) {
    if (curr_block->ptr == ptr) {
      curr_block->allocated = false;
      // Try to merge with previous block
      if (curr_block->prev && !curr_block->prev->allocated) {
        curr_block->prev->size += curr_block->size;
        curr_block->prev->next = curr_block->next;
        if (curr_block->next) {
          curr_block->next->prev = curr_block->prev;
        }
        curr_block = curr_block->prev;
      }
      // Try to merge with next block
      if (curr_block->next && !curr_block->next->allocated) {
        curr_block->size += curr_block->next->size;
        curr_block->next = curr_block->next->next;
        if (curr_block->next) {
          curr_block->next->prev = curr_block;
        }
      }
      return;
    }
    curr_block = curr_block->next;
  }
  HT_RUNTIME_ERROR << "Cannot find a memory block start with " << ptr << " in the memory manager";
}

} // namespace impl
} // namespace hetu