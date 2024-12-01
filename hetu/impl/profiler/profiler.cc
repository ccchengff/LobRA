#include "hetu/impl/profiler/profiler.h"

namespace hetu {
namespace impl {

std::once_flag Profile::_init_flag;
std::vector<std::shared_ptr<Profile>> Profile::_global_profile;
thread_local std::stack<ProfileId> Profile::_cur_profile_ctx;

ProfileId Profile::_next_profile_id() {
  static std::atomic<ProfileId> _global_profile_id{0};
  return _global_profile_id++;
}

void Profile::Init() {
  // exit handler
  auto status = std::atexit([]() {
    HT_LOG_DEBUG << "Clearing and destructing all profiler...";
    for (auto& profile : Profile::_global_profile) {
      if (profile == nullptr)
        continue;
      profile->Clear();
    }
    Profile::_global_profile.clear();
    HT_LOG_DEBUG << "Destructed all profiler";
  });
  HT_ASSERT(status == 0)
      << "Failed to register the exit function for memory pools.";

  auto concurrency = std::thread::hardware_concurrency();
  Profile::_global_profile.reserve(MIN(concurrency, 16) * 2);
}

} // namespace impl
} // namespace hetu