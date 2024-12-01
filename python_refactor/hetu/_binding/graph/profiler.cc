#include "hetu/_binding/graph/profiler.h"
#include "hetu/_binding/graph/graph.h"
#include "hetu/_binding/constants.h"
#include "hetu/_binding/utils/pybind_common.h"
#include "hetu/_binding/utils/except.h"
#include "hetu/_binding/utils/decl_utils.h"
#include "hetu/_binding/utils/arg_parser.h"
#include "hetu/impl/profiler/profiler.h"

namespace hetu {
namespace impl {

PyObject* PyProfile_New(ProfileId profile_id) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyProfile_Type->tp_alloc(PyProfile_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyProfile";
  auto* self = reinterpret_cast<PyProfile*>(unsafe_self);
  self->profile_id = profile_id;
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

void PyProfile_dealloc(PyProfile* self){
  hetu::impl::Profile::get_profile(self->profile_id)->~Profile();
  Py_TYPE(self)->tp_free(self);
}

PyObject* PyProfile_id(PyProfile* self) {
  HT_PY_FUNC_BEGIN
  return PyLong_FromInteger(self->profile_id);
  HT_PY_FUNC_END
}

PyObject* PyMakeNewProfile(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "make_new_profile(bool enabled, bool use_cpu, bool use_cuda, bool record_shapes, bool profile_memory)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return PyProfile_New(
      Profile::make_new_profile(parsed_args.get_bool(0), parsed_args.get_bool(1), 
       parsed_args.get_bool(2), parsed_args.get_bool(3), parsed_args.get_bool(4)).id());
    Py_RETURN_NONE;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyPushProfileCtx(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "push_profile_ctx(int profile_id)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    Profile::push_profile_ctx(parsed_args.get_int64(0));
    Py_RETURN_NONE;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyPopProfileCtx(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "pop_op_ctx()"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    Profile::pop_profile_ctx();
    Py_RETURN_NONE;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyMethodDef PyProfileCtx_methods[] = {
  {"make_new_profile", (PyCFunction) PyMakeNewProfile,  METH_VARARGS | METH_KEYWORDS, nullptr},
  {"push_profile_ctx", (PyCFunction) PyPushProfileCtx,  METH_VARARGS | METH_KEYWORDS, nullptr},
  {"pop_profile_ctx", (PyCFunction) PyPopProfileCtx,  METH_VARARGS | METH_KEYWORDS, nullptr},
  {nullptr}
};

// NOLINTNEXTLINE
PyGetSetDef PyProfile_properties[] = {
  {PY_GET_SET_DEF_NAME("id"), (getter) PyProfile_id, nullptr, nullptr, nullptr}, 
  {nullptr}
};

PyObject* PyProfileSummary(PyProfile* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "summary()",
  });

  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    auto profiler = Profile::get_profile(self->profile_id);
    PyObject* py_dict = PyDict_New();
    if (!py_dict)
      return nullptr;

    auto op_view = profiler->get_op_view();
    PyObject* py_list_op_view = PyList_New((int)op_view.size());
    PyObject* tmp_list;

    for (int i = 0; i < op_view.size(); i++) {
	    auto op_record = op_view[i];
      tmp_list = PyList_New(3);
      PyObject* name = PyUnicode_FromString(op_record.name);
      PyObject* type = PyUnicode_FromString(op_record.type);
	    PyObject* total_time = PyFloat_FromDouble(op_record.cost_time);
	    auto inputs_shape_c = op_record.inputs_shape;
      PyList_SET_ITEM(tmp_list, 0, name);
      PyList_SET_ITEM(tmp_list, 1, type);
      PyList_SET_ITEM(tmp_list, 2, total_time);
	    PyList_SET_ITEM(py_list_op_view, i, tmp_list);
    }
    
    PyDict_SetItemString(py_dict, "op_view", py_list_op_view);
    Py_DECREF(py_list_op_view);
    
    auto optype_view = profiler->get_optype_view();
    PyObject* py_list_optype_view = PyList_New((int)optype_view.size());
    for (int i = 0; i < optype_view.size(); i++) {
      auto op_record = optype_view[i];
      tmp_list = PyList_New(4);
      PyObject* type = PyUnicode_FromString(op_record.first);
      PyObject* total_time = PyFloat_FromDouble(op_record.second.first);
      PyObject* avg_time = PyFloat_FromDouble(op_record.second.second.first);
      PyObject* cnt = PyLong_FromLong(op_record.second.second.second);
      PyList_SET_ITEM(tmp_list, 0, type);
      PyList_SET_ITEM(tmp_list, 1, total_time);
      PyList_SET_ITEM(tmp_list, 2, avg_time);
      PyList_SET_ITEM(tmp_list, 3, cnt);
      PyList_SET_ITEM(py_list_optype_view, i, tmp_list);
    }

    PyDict_SetItemString(py_dict, "optype_view", py_list_optype_view);
    Py_DECREF(py_list_optype_view);
    if (profiler->record_shapes()) {
      auto optype_with_inputs_view = profiler->get_optype_with_inputs_view();
      PyObject *py_list_optype_with_inputs_view;
      py_list_optype_with_inputs_view = PyList_New((int)optype_with_inputs_view.size());

      for (int i = 0; i < optype_with_inputs_view.size(); i++) {
        tmp_list = PyList_New(5);
        auto op_record = optype_with_inputs_view[i];
        PyObject* type = PyUnicode_FromString(op_record.first.first);
        PyObject* total_time = PyFloat_FromDouble(op_record.second.first);
        PyObject* avg_time = PyFloat_FromDouble(op_record.second.second.first);
        PyObject* cnt = PyLong_FromLong(op_record.second.second.second);
        auto inputs_shape_c = op_record.first.second;
        PyObject* inputs_shape = PyTuple_New(inputs_shape_c.size());

        for (int i = 0; i < inputs_shape_c.size(); i++) {
          auto shape = inputs_shape_c[i];
          PyObject* tuple_tmp = PyTuple_New(shape.size());
          for (int j = 0; j < shape.size(); j++) {
            PyTuple_SetItem(tuple_tmp, j, PyLong_FromLong(shape[j]));
          }
          PyTuple_SetItem(inputs_shape, i, tuple_tmp);
        }

        PyList_SET_ITEM(tmp_list, 0, type);
        PyList_SET_ITEM(tmp_list, 1, inputs_shape);
        PyList_SET_ITEM(tmp_list, 2, total_time);
        PyList_SET_ITEM(tmp_list, 3, avg_time);
        PyList_SET_ITEM(tmp_list, 4, cnt);
        PyList_SET_ITEM(py_list_optype_with_inputs_view, i, tmp_list);
      }
      PyDict_SetItemString(py_dict, "optype_with_inputs_view", py_list_optype_with_inputs_view);
      Py_DECREF(py_list_optype_with_inputs_view);
    }
    auto graph_view = profiler->get_graph_view();
    if(graph_view.size() == 0)
      return py_dict;
    PyObject* py_list_graph_view = PyList_New((int)graph_view.size());
    for (int i = 0; i < graph_view.size(); i++) {
      auto record = graph_view[i];
      tmp_list = PyList_New(2);
      PyObject* type = PyUnicode_FromString(record.first);
      PyObject* total_time = PyFloat_FromDouble(record.second);
      PyList_SET_ITEM(tmp_list, 0, type);
      PyList_SET_ITEM(tmp_list, 1, total_time);
      PyList_SET_ITEM(py_list_graph_view, i, tmp_list);
    }
    PyDict_SetItemString(py_dict, "graph_view", py_list_graph_view);
    Py_DECREF(py_list_graph_view);
    return py_dict;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

// NOLINTNEXTLINE
PyMethodDef PyProfile_methods[] = {
  {"summary", (PyCFunction) PyProfileSummary,  METH_VARARGS | METH_KEYWORDS, nullptr},
  {nullptr}
};

// NOLINTNEXTLINE
PyTypeObject PyProfile_Type_obj = {
  PyVarObject_HEAD_INIT(nullptr, 0) 
  "hetu.Profile", /* tp_name */
  sizeof(PyProfile), /* tp_basicsize */
  0, /* tp_itemsize */
  (destructor) PyProfile_dealloc, /* tp_dealloc */
  0, /* tp_vectorcall_offset */
  nullptr, /* tp_getattr */
  nullptr, /* tp_setattr */
  nullptr, /* tp_reserved */
  nullptr, /* tp_repr */
  nullptr, /* tp_as_number */
  nullptr, /* tp_as_sequence */
  nullptr, /* tp_as_mapping */
  nullptr, /* tp_hash  */
  nullptr, /* tp_call */
  nullptr, /* tp_str */
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
  PyProfile_methods, /* tp_methods */
  nullptr, /* tp_members */
  PyProfile_properties, /* tp_getset */
  nullptr, /* tp_base */
  nullptr, /* tp_dict */
  nullptr, /* tp_descr_get */
  nullptr, /* tp_descr_set */
  0, /* tp_dictoffset */
  nullptr, /* tp_init */
  nullptr, /* tp_alloc */
  nullptr, /* tp_new */
};

PyTypeObject* PyProfile_Type = &PyProfile_Type_obj;

void AddPyProfileTypeToModule(py::module_& module) {
  HT_RUNTIME_ERROR_IF(PyType_Ready(PyProfile_Type) < 0) 
    << "PyProfile_Type not ready";
  Py_INCREF(PyProfile_Type);
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddObject(
      module.ptr(), "Profile", reinterpret_cast<PyObject*>(PyProfile_Type)))
    << "Failed to add PyProfile_Type";
}

void AddProfileContextManagingFunctionsToModule(py::module_& m) {
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddFunctions(m.ptr(), PyProfileCtx_methods))
    << "Failed to add profile context managing methods";
}

} // namespace impl
} // namespace hetu

