// SPDX-License-Identifier: Apache-2.0
//
// Typed row-major views over NumPy buffers handed in once at kernel construction.
// Each row is contiguous; rows may be further apart than their width (MJBatch >= 0.1.4
// binds time, qpos, qvel, ctrl and act as views into the state rows, nstate elements
// apart). Index rows with Row(r) and the elements of a 1-D array with [i], never p[i].
// A kernel keeps a strong reference to every array it views, so the buffers (and,
// for MJBatch bind() views, the owning Batch) outlive the kernel. Kernels never
// allocate per call; all per-slot state lives in these Python-owned arrays, which
// makes snapshot/restore a plain row copy on the Python side.
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace rsb {

// The trailing axes of each row are C-contiguous (numpy's rule: an extent-1 axis may carry
// any stride and a size-0 array is contiguous); the leading axis may carry any stride.
inline bool RowsContig(const nb::ndarray<>& a) {
  if (a.size() == 0) return true;
  int64_t expect = 1;
  for (size_t i = a.ndim(); i-- > 1;) {
    if (a.shape(i) != 1 && a.stride(i) != expect) return false;
    expect *= static_cast<int64_t>(a.shape(i));
  }
  return true;
}

template <class T>
struct Mat {
  nb::object keep;
  T* p = nullptr;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t ld = 0;  // elements from one row to the next (>= cols)
  T* Row(int64_t r) const { return p + r * ld; }
  T& operator[](int64_t i) const { return p[i * ld]; }  // element i of a one-column array
  bool Valid() const { return p != nullptr; }
};

template <class T>
Mat<T> ToMat(nb::handle h, const std::string& name, int64_t rows = -1, int64_t cols = -1) {
  Mat<T> m;
  if (!h.is_valid() || h.is_none()) return m;
  nb::ndarray<> a;
  if (!nb::try_cast(h, a, false)) throw nb::type_error((name + ": not an array").c_str());
  if (a.dtype() != nb::dtype<T>()) throw nb::type_error((name + ": wrong dtype").c_str());
  if (a.device_type() != nb::device::cpu::value) throw nb::value_error((name + ": must be a CPU array").c_str());
  if (a.ndim() < 1) throw nb::value_error((name + ": must have a leading axis").c_str());
  if (!RowsContig(a)) throw nb::value_error((name + ": each row must be contiguous").c_str());
  int64_t c = 1;
  for (size_t i = 1; i < a.ndim(); ++i) c *= static_cast<int64_t>(a.shape(i));
  const int64_t r = static_cast<int64_t>(a.shape(0));
  const int64_t ld = r > 1 && c > 0 ? static_cast<int64_t>(a.stride(0)) : c;
  if (ld < c) throw nb::value_error((name + ": rows overlap or run backwards").c_str());
  if (rows >= 0 && r != rows) throw nb::value_error((name + ": wrong row count").c_str());
  if (cols >= 0 && c != cols) throw nb::value_error((name + ": wrong row width").c_str());
  m.keep = nb::borrow(h);
  m.p = static_cast<T*>(a.data());
  m.rows = r;
  m.cols = c;
  m.ld = ld;
  return m;
}

template <class T>
Mat<T> Get(const nb::dict& arrays, const char* key, int64_t rows = -1, int64_t cols = -1,
           bool required = true) {
  if (!arrays.contains(key)) {
    if (required) throw nb::key_error((std::string("missing array ") + key).c_str());
    return Mat<T>{};
  }
  Mat<T> m = ToMat<T>(arrays[key], key, rows, cols);
  if (required && !m.Valid()) throw nb::value_error((std::string("array ") + key + " is None").c_str());
  return m;
}

template <class T>
T Scalar(const nb::dict& cfg, const char* key) {
  if (!cfg.contains(key)) throw nb::key_error((std::string("missing config ") + key).c_str());
  return nb::cast<T>(cfg[key]);
}

using Slots = nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

inline void CheckSlots(const Slots& slots, int64_t n) {
  const int32_t* p = slots.data();
  for (size_t j = 0; j < slots.shape(0); ++j) {
    if (p[j] < 0 || p[j] >= n) throw nb::index_error("slot out of range");
  }
}

}  // namespace rsb
