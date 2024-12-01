#pragma once

#include <Python.h>
#include "hetu/impl/communication/comm_group.h"
#include "hetu/_binding/utils/pybind_common.h"

namespace hetu {

struct PyCommGroup {
  PyObject_HEAD;
};

extern PyTypeObject* PyCommGroup_Type;

void AddPyCommGroupTypeToModule(py::module_& module);

} // namespace hetu
