#pragma once

#include <Python.h>
#include "hetu/graph/distributed_states.h"
#include "hetu/_binding/utils/pybind_common.h"

namespace hetu {
namespace graph {

struct PyDistributedStates {
  PyObject_HEAD;
  DistributedStates distributed_states;
};

extern PyTypeObject* PyDistributedStates_Type;

inline bool PyDistributedStates_Check(PyObject* obj) {
  return PyDistributedStates_Type && PyObject_TypeCheck(obj, PyDistributedStates_Type);
}

inline bool PyDistributedStates_CheckExact(PyObject* obj) {
  return PyDistributedStates_Type && obj->ob_type == PyDistributedStates_Type;
}

PyObject* PyDistributedStates_New(const DistributedStates& ds);

PyObject* PyDistributedStatesList_New(const DistributedStatesList& multi_ds);

void AddPyDistributedStatesTypeToModule(py::module_& module);

/******************************************************
 * ArgParser Utils
 ******************************************************/

inline bool CheckPyDistributedStates(PyObject* obj) {
  return PyDistributedStates_Check(obj);
}

inline DistributedStates DistributedStates_FromPyObject(PyObject* obj) {
  return reinterpret_cast<PyDistributedStates*>(obj)->distributed_states;
}

inline bool CheckPyDistributedStatesList(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  if (is_tuple || PyList_Check(obj)) {
    size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
    if (size > 0) {
      // only check for the first item for efficiency
      auto* item = is_tuple ? PyTuple_GET_ITEM(obj, 0) \
                            : PyList_GET_ITEM(obj, 0);
      if (!CheckPyDistributedStates(item))
        return false;
    }
    return true;
  }
  return false;
}

inline DistributedStatesList DistributedStatesList_FromPyObject(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  DistributedStatesList ret(size);
  for (size_t i = 0; i < size; i++) {
    auto* item = is_tuple ? PyTuple_GET_ITEM(obj, i) : PyList_GET_ITEM(obj, i);
    ret[i] = DistributedStates_FromPyObject(item);
  }
  return ret;
}

} // namespace graph
} // namespace hetu
