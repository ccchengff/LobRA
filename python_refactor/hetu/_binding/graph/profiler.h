#pragma once

#include <Python.h>
#include "hetu/_binding/core/ndarray.h"
#include "hetu/_binding/graph/tensor.h"
#include "hetu/_binding/utils/numpy.h"
#include "hetu/_binding/utils/pybind_common.h"
#include "hetu/impl/profiler/profiler.h"

namespace hetu {
namespace impl {

struct PyProfile {
  PyObject_HEAD;
  ProfileId profile_id;
};

extern PyTypeObject* PyProfile_Type;

PyObject* PyProfile_New(ProfileId profile_id);

void AddPyProfileTypeToModule(py::module_& module);
void AddProfileContextManagingFunctionsToModule(py::module_& m);

} // namespace impl
} // namespace hetu
