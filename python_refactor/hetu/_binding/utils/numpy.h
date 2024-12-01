#pragma once

#include <Python.h>
#include "hetu/core/ndarray.h"

// Note: Do NOT include `numpy/arrayobject.h` in this header 
// to get rid of the `NO_IMPORT_ARRAY` definition 
// (see https://docs.scipy.org/doc/numpy-1.17.0/reference/c-api.array.html#miscellaneous
// for more details). For simplicity, we put all numpy-related calls 
// to a single .c file, i.e., `numpy.cc`.

namespace hetu {

bool CheckNumpyInt(PyObject* obj);

bool CheckNumpyFloat(PyObject* obj);

bool CheckNumpyBool(PyObject* obj);

bool CheckNumpyArray(PyObject* obj);

bool CheckNumpyArrayList(PyObject* obj);

DataType GetNumpyArrayDataType(PyObject* obj);

NDArray NDArrayFromNumpy(PyObject* obj, const HTShape& dynamic_shape = {}, DataType datatype = kUndeterminedDataType);

NDArrayList NDArrayListFromNumpyList(PyObject* obj, const HTShape& dynamic_shape = {}, DataType datatype = kUndeterminedDataType);

PyObject* NDArrayToNumpy(NDArray ndarray, bool force, bool save = false);

PyObject* NumpyFromSequences(PyObject* obj);

} // namespace hetu
