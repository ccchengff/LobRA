#include "hetu/_binding/graph/recompute.h"
#include "hetu/_binding/constants.h"
#include "hetu/_binding/utils/pybind_common.h"
#include "hetu/_binding/utils/except.h"
#include "hetu/_binding/utils/decl_utils.h"
#include "hetu/_binding/utils/arg_parser.h"

namespace hetu {
namespace graph {

PyObject* PyPushRecomputeCtx(PyObject*) {
  HT_PY_FUNC_BEGIN
  Recompute::set_recompute_enabled();
  Py_RETURN_NONE;
  HT_PY_FUNC_END
}

PyObject* PyPopRecomputeCtx(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  Recompute::reset_recompute_enabled();
  Py_RETURN_NONE;
  HT_PY_FUNC_END
}

// NOLINTNEXTLINE
PyMethodDef PyRecomputeCtx_methods[] = { 
  {"push_recompute_ctx", (PyCFunction) PyPushRecomputeCtx, METH_NOARGS, nullptr},
  {"pop_recompute_ctx", (PyCFunction) PyPopRecomputeCtx, METH_NOARGS, nullptr},
  {nullptr}
};

void AddRecomputeContextManagingFunctionsToModule(py::module_& m) {
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddFunctions(m.ptr(), PyRecomputeCtx_methods))
    << "Failed to add recompute context managing methods";
}

} // namespace graph
} // namespace hetu
