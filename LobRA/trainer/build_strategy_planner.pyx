import cython
from libcpp.vector cimport vector

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def build_strategy_planner_cython(int gpu_num, int strategy_pool_size, int[:] strategy_pool):
    cdef int i, j, strategy_gpu
    cdef vector[int] dp = vector[int](gpu_num + 1, 0)
    dp[0] = 1  # 初始化 DP
    cdef vector[vector[vector[int]]] paths = vector[vector[vector[int]]](gpu_num + 1)
    paths[0].push_back(vector[int]())  # 初始化路径

    # 动态规划
    for i in range(strategy_pool_size):  # 遍历每个策略
        strategy_gpu = strategy_pool[i]
        for j in range(strategy_gpu, gpu_num + 1):  # 从后往前遍历
            if dp[j - strategy_gpu] > 0:
                dp[j] += dp[j - strategy_gpu]
                # 更新路径
                update_paths(paths, j, j - strategy_gpu, i)

    # 将结果转换回 Python 格式以便返回
    cdef list py_paths = []
    for path in paths[gpu_num]:
        py_paths.append([p for p in path])

    # 返回结果
    return dp[gpu_num], py_paths

cdef void update_paths(vector[vector[vector[int]]] &paths, int target, int source, int strategy_index):
    cdef vector[int] new_path
    for path in paths[source]:
        new_path = vector[int](path)  # 使用拷贝构造函数
        new_path.push_back(strategy_index)
        paths[target].push_back(new_path)
