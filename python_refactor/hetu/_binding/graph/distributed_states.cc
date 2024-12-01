#include "hetu/_binding/graph/distributed_states.h"
#include "hetu/_binding/core/device.h"
#include "hetu/_binding/utils/except.h"
#include "hetu/_binding/utils/arg_parser.h"
#include "hetu/_binding/utils/decl_utils.h"
#include "hetu/_binding/utils/function_registry.h"

namespace hetu {
namespace graph {

PyObject* PyDistributedStates_New(const DistributedStates& ds) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyDistributedStates_Type->tp_alloc(PyDistributedStates_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyDistributedStates";
  auto* self = reinterpret_cast<PyDistributedStates*>(unsafe_self);
  new (&self->distributed_states) DistributedStates(ds);
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END  
}

PyObject* PyDistributedStatesList_New(const DistributedStatesList& multi_ds) {
  HT_PY_FUNC_BEGIN
  PyObject* ret = PyList_New(multi_ds.size());
  HT_RUNTIME_ERROR_IF(!ret) << "Failed to alloc list";
  for (size_t i = 0; i < multi_ds.size(); i++) {
    auto* distributed_states_obj = PyDistributedStates_New(multi_ds[i]);
    PyList_SET_ITEM(ret, i, distributed_states_obj);
  }
  return ret;
  HT_PY_FUNC_END
}

PyObject* PyDistributedStates_pynew(PyTypeObject* type, PyObject* args, 
                                    PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyDistributedStates_Type->tp_alloc(PyDistributedStates_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyDistributedStates";  
  auto* self = reinterpret_cast<PyDistributedStates*>(unsafe_self);
  static PyArgParser parser({
    "DistributedStates(int device_num, std::unordered_map<int,int> states, std::vector<int64_t> order=None, bool zero=False)",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    int32_t device_num = parsed_args.get_int64(0);
    std::unordered_map<int32_t, int32_t> states = parsed_args.get_unordered_map(1);
    std::vector<int32_t> order;
    if (parsed_args.has(2)) {
      std::vector<int64_t> int64_list = parsed_args.get_int64_list(2);
      for (auto o : int64_list) {
        order.push_back(static_cast<int32_t>(o));
      }
    }
    bool zero = parsed_args.get_bool_or_default(3);
    new (&self->distributed_states) DistributedStates(device_num, states, order, zero);
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

void PyDistributedStates_dealloc(PyDistributedStates* self) {
  (&self->distributed_states)->~DistributedStates();
  Py_TYPE(self)->tp_free(self);
}

PyObject* PyDistributedStates_str(PyDistributedStates* self) {
  HT_PY_FUNC_BEGIN
  return PyUnicode_FromString(self->distributed_states.ds_info());
  HT_PY_FUNC_END
}

PyObject* PyDistributedStates_repr(PyDistributedStates* self) {
  return PyDistributedStates_str(self);
}

PyObject* PyDistributedStates_device_num(PyDistributedStates* self) {
  HT_PY_FUNC_BEGIN
  return PyLong_FromInteger(self->distributed_states.get_device_num()); 
  HT_PY_FUNC_END
}

PyObject* PyDistributedStates_states(PyDistributedStates* self) {
  HT_PY_FUNC_BEGIN
  return PyDict_FromUnorderedMap(self->distributed_states.get_states()); 
  HT_PY_FUNC_END
}

PyObject* PyDistributedStates_order(PyDistributedStates* self) {
  HT_PY_FUNC_BEGIN
  return PyLongList_FromIntegerList(self->distributed_states.get_order());
  HT_PY_FUNC_END
}

/*
// deprecated
PyObject* PyDistributedStates_placement_group(PyDistributedStates* self) {
  HT_PY_FUNC_BEGIN
  return PyDeviceGroup_New(self->distributed_states.get_placement_group());
  HT_PY_FUNC_END
}
*/

PyObject* PyDistributedStates_zero(PyDistributedStates* self) {
  HT_PY_FUNC_BEGIN
  Py_RETURN_BOOLEAN_COND(self->distributed_states.zero());
  HT_PY_FUNC_END
}

PyObject* PyDistributedStates_is_pure_duplicate(PyDistributedStates* self) {
  HT_PY_FUNC_BEGIN
  Py_RETURN_BOOLEAN_COND(self->distributed_states.check_pure_duplicate());
  HT_PY_FUNC_END
}

PyObject* PyDistributedStates_check_equal(PyDistributedStates* self, PyObject* args) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({"check_equal(DistributedStates ds)"});
  auto parsed_args = parser.parse(args, nullptr);
  if (parsed_args.signature_index() == 0) {
    DistributedStates ds = parsed_args.get_distributed_states(0);
    Py_RETURN_BOOLEAN_COND(self->distributed_states.check_equal(ds));
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }  
  HT_PY_FUNC_END
}

PyObject* PyDistributedStates_get_dim(PyDistributedStates* self, PyObject* args) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({"get_dim(int dim)"});
  auto parsed_args = parser.parse(args, nullptr);
  if (parsed_args.signature_index() == 0) {
    return PyLong_FromInteger(self->distributed_states.get_dim(parsed_args.get_int64(0)));
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }  
  HT_PY_FUNC_END
}

PyObject* PyDistributedStates_map_to_local_data(PyDistributedStates* self, PyObject* args) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({"map_to_local_data(DistributedStates ds, int device_index)"});
  auto parsed_args = parser.parse(args, nullptr);
  if (parsed_args.signature_index() == 0) {
    DistributedStates ds = parsed_args.get_distributed_states(0);
    int device_index = parsed_args.get_int64(1);
    return PyDict_FromUnorderedMap(ds.map_device_to_state_index(device_index));
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }  
  HT_PY_FUNC_END
}

PyObject* PyDistributedStates_get_dup_group_index(PyDistributedStates* self, PyObject* args) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({"get_dup_group_index(int device_index)"});
  auto parsed_args = parser.parse(args, nullptr);
  if (parsed_args.signature_index() == 0) {
    return PyLong_FromInteger(self->distributed_states.get_dup_group_index(parsed_args.get_int64(0)));
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }  
  HT_PY_FUNC_END
}

// NOLINTNEXTLINE
PyGetSetDef PyDistributedStates_properties[] = {
  {PY_GET_SET_DEF_NAME("device_num"), (getter) PyDistributedStates_device_num, nullptr, nullptr, nullptr},
  {PY_GET_SET_DEF_NAME("states"), (getter) PyDistributedStates_states, nullptr, nullptr, nullptr},
  {PY_GET_SET_DEF_NAME("order"), (getter) PyDistributedStates_order, nullptr, nullptr, nullptr},
  // {PY_GET_SET_DEF_NAME("placement_group"), (getter) PyDistributedStates_placement_group, nullptr, nullptr, nullptr},
  {PY_GET_SET_DEF_NAME("zero"), (getter) PyDistributedStates_zero, nullptr, nullptr, nullptr},
  {PY_GET_SET_DEF_NAME("is_pure_duplicate"), (getter) PyDistributedStates_is_pure_duplicate, nullptr, nullptr, nullptr},
  {nullptr}
};

// NOLINTNEXTLINE
PyMethodDef PyDistributedStates_methods[] = {
  {"check_equal", (PyCFunction) PyDistributedStates_check_equal, METH_VARARGS, nullptr },
  {"get_dim", (PyCFunction) PyDistributedStates_get_dim, METH_VARARGS, nullptr },
  {"get_dup_group_index", (PyCFunction) PyDistributedStates_get_dup_group_index, METH_VARARGS, nullptr },
  {nullptr}
};

// NOLINTNEXTLINE
PyTypeObject PyDistributedStates_Type_obj = {
  PyVarObject_HEAD_INIT(nullptr, 0) 
  "hetu.DistributedStates", /* tp_name */
  sizeof(PyDistributedStates), /* tp_basicsize */
  0, /* tp_itemsize */
  (destructor) PyDistributedStates_dealloc, /* tp_dealloc */
  0, /* tp_vectorcall_offset */
  nullptr, /* tp_getattr */
  nullptr, /* tp_setattr */
  nullptr, /* tp_reserved */
  (reprfunc) PyDistributedStates_repr, /* tp_repr */
  nullptr, /* tp_as_number */
  nullptr, /* tp_as_sequence */
  nullptr, /* tp_as_mapping */
  nullptr, /* tp_hash  */
  nullptr, /* tp_call */
  (reprfunc) PyDistributedStates_str, /* tp_str */
  nullptr, /* tp_getattro */
  nullptr, /* tp_setattro */
  nullptr, /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT, /* tp_flags */
  nullptr, /* tp_doc */
  nullptr, /* tp_traverse */
  nullptr, /* tp_clear */
  nullptr, /* tp_richcompare */
  0, /* tp_weaklistoffset */
  nullptr, /* tp_iter */
  nullptr, /* tp_iternext */
  PyDistributedStates_methods, /* tp_methods */
  nullptr, /* tp_members */
  PyDistributedStates_properties, /* tp_getset */
  nullptr, /* tp_base */
  nullptr, /* tp_dict */
  nullptr, /* tp_descr_get */
  nullptr, /* tp_descr_set */
  0, /* tp_dictoffset */
  nullptr, /* tp_init */
  nullptr, /* tp_alloc */
  PyDistributedStates_pynew, /* tp_new */
};
PyTypeObject* PyDistributedStates_Type = &PyDistributedStates_Type_obj;

std::vector<PyMethodDef> InitDistributedStatesPyClassMethodDefs() {
  std::vector<PyMethodDef> ret = {{nullptr}};
  AddPyMethodDefs(ret, {
    {"map_to_local_data", (PyCFunction) PyDistributedStates_map_to_local_data, METH_VARARGS | METH_KEYWORDS, nullptr },
    {nullptr}
  });
  AddPyMethodDefs(ret, hetu::graph::get_registered_tensor_class_methods()); // TODO: register distributed states class methods
  return ret;
}

void AddPyDistributedStatesTypeToModule(py::module_& module) {
  HT_RUNTIME_ERROR_IF(PyType_Ready(PyDistributedStates_Type) < 0) 
    << "PyDistributedStates_Type not ready";
  Py_INCREF(PyDistributedStates_Type);
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddObject(
      module.ptr(), "DistributedStates", 
      reinterpret_cast<PyObject*>(PyDistributedStates_Type)))
    << "Failed to add PyDistributedStates_Type";

  static auto tensor_class_methods = InitDistributedStatesPyClassMethodDefs();
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddFunctions(
      module.ptr(), tensor_class_methods.data()))
    << "Failed to add Tensor class methods";     
}

} // namespace graph
} // namespace hetu
