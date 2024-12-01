#pragma once

#include <Python.h>
#include "hetu/graph/offload/activation_cpu_offload.h"
#include "hetu/_binding/core/ndarray.h"
#include "hetu/_binding/graph/tensor.h"
#include "hetu/_binding/utils/numpy.h"
#include "hetu/_binding/utils/pybind_common.h"

namespace hetu {
namespace graph {

/******************************************************
 * For contextlib usage
 ******************************************************/

void AddCPUOffloadContextManagingFunctionsToModule(py::module_&);

} // namespace graph
} // namespace hetu
