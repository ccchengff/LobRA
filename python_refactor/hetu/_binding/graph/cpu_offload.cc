#include "hetu/_binding/graph/cpu_offload.h"
#include "hetu/_binding/constants.h"
#include "hetu/_binding/utils/pybind_common.h"
#include "hetu/_binding/utils/except.h"
#include "hetu/_binding/utils/decl_utils.h"
#include "hetu/_binding/utils/arg_parser.h"

namespace hetu {
namespace graph {

PyObject* PyPushCPUOffloadCtx(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  ActivationCPUOffload::set_cpu_offload_enabled();
  Py_RETURN_NONE;
  HT_PY_FUNC_END
}

PyObject* PyPopCPUOffloadCtx(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  ActivationCPUOffload::reset_cpu_offload_enabled();
  Py_RETURN_NONE;
  HT_PY_FUNC_END
}

// NOLINTNEXTLINE
PyMethodDef PyCPUOffloadCtx_methods[] = { 
  {"push_cpu_offload_ctx", (PyCFunction) PyPushCPUOffloadCtx, METH_NOARGS, nullptr},
  {"pop_cpu_offload_ctx", (PyCFunction) PyPopCPUOffloadCtx, METH_NOARGS, nullptr},
  {nullptr}
};

void AddCPUOffloadContextManagingFunctionsToModule(py::module_& m) {
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddFunctions(m.ptr(), PyCPUOffloadCtx_methods))
    << "Failed to add cpu offload context managing methods";
}

} // namespace graph
} // namespace hetu
