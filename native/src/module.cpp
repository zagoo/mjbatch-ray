// SPDX-License-Identifier: Apache-2.0
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include "kernels.h"
#include "philox.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using KeyArray = nb::ndarray<const uint64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using F64Out = nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

// out[i, k] = draw(keys[i], seq, index0 + k, stream) for each row i.
void PhiloxFill(const KeyArray& keys, uint64_t seq, uint64_t index0, uint64_t stream, F64Out out,
                bool normal) {
  if (keys.shape(1) != 2 || out.shape(0) != keys.shape(0)) throw nb::value_error("keys/out shape");
  nb::gil_scoped_release release;
  const size_t n = keys.shape(0), k = out.shape(1);
  for (size_t i = 0; i < n; ++i) {
    const uint64_t* key = keys.data() + 2 * i;
    for (size_t j = 0; j < k; ++j) {
      out.data()[i * k + j] = normal ? rsb::Normal(key, seq, index0 + j, stream)
                                     : rsb::Uniform(key, seq, index0 + j, stream);
    }
  }
}

nb::tuple TaskSpec(const std::string& name, int version) {
  if (version != 1) throw nb::value_error("unsupported task kernel version");
  if (name == "balance") return nb::make_tuple(0, 3, 1);
  if (name == "velocity_tracking") return nb::make_tuple(3, 4, 3);
  if (name == "reach") return nb::make_tuple(3, 3, 1);
  throw nb::value_error(("unknown task kernel " + name).c_str());
}

}  // namespace

NB_MODULE(_native, m) {
  m.attr("ABI_VERSION") = 1;
  m.attr("TASK_METRICS") = rsb::kTaskMetrics;

  m.def("philox_block", [](uint64_t k0, uint64_t k1, uint64_t c0, uint64_t c1, uint64_t c2, uint64_t c3) {
    const uint64_t key[2] = {k0, k1};
    const uint64_t ctr[4] = {c0, c1, c2, c3};
    const rsb::Block b = rsb::Philox4x64(ctr, key);
    return nb::make_tuple(b.v[0], b.v[1], b.v[2], b.v[3]);
  });
  m.def("philox_uniform", [](const KeyArray& keys, uint64_t seq, uint64_t index0, uint64_t stream, F64Out out) {
    PhiloxFill(keys, seq, index0, stream, out, false);
  }, "keys"_a.noconvert(), "seq"_a, "index0"_a, "stream"_a, "out"_a.noconvert());
  m.def("philox_normal", [](const KeyArray& keys, uint64_t seq, uint64_t index0, uint64_t stream, F64Out out) {
    PhiloxFill(keys, seq, index0, stream, out, true);
  }, "keys"_a.noconvert(), "seq"_a, "index0"_a, "stream"_a, "out"_a.noconvert());
  m.def("task_spec", &TaskSpec, "name"_a, "version"_a,
        "(num_commands, num_state, num_components) of a registered task kernel");

  nb::class_<rsb::ActionProcessor>(m, "ActionProcessor")
      .def(nb::init<const nb::dict&, const nb::dict&>(), "config"_a, "arrays"_a)
      .def("reset", &rsb::ActionProcessor::Reset, "slots"_a.noconvert())
      .def("process", &rsb::ActionProcessor::Process, "slots"_a.noconvert(), "actions"_a.noconvert());

  nb::class_<rsb::Controller>(m, "Controller")
      .def(nb::init<const nb::dict&, const nb::dict&>(), "config"_a, "arrays"_a)
      .def("reset", &rsb::Controller::Reset, "slots"_a.noconvert())
      .def("before_substep", &rsb::Controller::BeforeSubstep, "slots"_a.noconvert(), "dt"_a)
      .def("neutralize", &rsb::Controller::Neutralize, "slots"_a.noconvert())
      .def("prepare_fused", &rsb::Controller::PrepareFused, "slots"_a.noconvert())
      .def("finish_fused", &rsb::Controller::FinishFused, "slots"_a.noconvert(), "rows"_a.noconvert(),
           "history"_a.noconvert(), "qpos_offset"_a, "qvel_offset"_a, "dt"_a)
      .def_prop_ro("fusable", &rsb::Controller::Fusable);

  nb::class_<rsb::Health>(m, "Health")
      .def(nb::init<const nb::dict&, const nb::dict&>(), "config"_a, "arrays"_a)
      .def("rebase", &rsb::Health::Rebase, "slots"_a.noconvert())
      .def("check", &rsb::Health::Check, "slots"_a.noconvert(), "dt"_a, "fault"_a.noconvert(),
           "warn_delta"_a.noconvert());

  nb::class_<rsb::Book>(m, "Book")
      .def(nb::init<const nb::dict&, const nb::dict&>(), "config"_a, "arrays"_a)
      .def("ready", &rsb::Book::Ready, "slots"_a.noconvert())
      .def("begin", &rsb::Book::Begin, "slots"_a.noconvert())
      .def("account", &rsb::Book::Account, "slots"_a.noconvert(), "status"_a.noconvert(), "completed"_a.noconvert(),
           "warn"_a.noconvert(), "ended"_a.noconvert(), "seq_out"_a.noconvert(), "good"_a.noconvert())
      .def("settle", &rsb::Book::Settle, "slots"_a.noconvert(), "status"_a.noconvert(), "good"_a.noconvert(),
           "reward"_a.noconvert(), "components"_a.noconvert(), "ended"_a.noconvert(), "flags"_a.noconvert())
      .def("fused_settle", &rsb::Book::FusedSettle, "controller"_a, "health"_a, "task"_a, "slots"_a.noconvert(),
           "failed"_a.noconvert(), "history"_a.noconvert(), "qpos_offset"_a, "qvel_offset"_a, "dt"_a,
           "timestep_ns"_a, "status"_a.noconvert(), "fault"_a.noconvert(), "completed"_a.noconvert(),
           "failed_substep"_a.noconvert(), "warn"_a.noconvert(), "ended"_a.noconvert(), "first"_a.noconvert(),
           "replay"_a.noconvert())
      .def("substep_settle", &rsb::Book::SubstepSettle, "health"_a, "task"_a, "active"_a.noconvert(),
           "pos"_a.noconvert(), "k"_a, "last"_a, "each_substep"_a, "dt"_a, "timestep_ns"_a, "status"_a.noconvert(),
           "fault"_a.noconvert(), "completed"_a.noconvert(), "failed_substep"_a.noconvert(), "warn"_a.noconvert(),
           "ended"_a.noconvert());

  nb::class_<rsb::Observation>(m, "Observation")
      .def(nb::init<const nb::dict&, const nb::dict&>(), "config"_a, "arrays"_a)
      .def("reset", &rsb::Observation::Reset, "slots"_a.noconvert())
      .def("observe", &rsb::Observation::Observe, "slots"_a.noconvert(), "out"_a.noconvert())
      .def_prop_ro("obs_dim", &rsb::Observation::obs_dim);

  nb::class_<rsb::TaskKernel>(m, "TaskKernel")
      .def(nb::init<const std::string&, int, const nb::dict&>(), "name"_a, "version"_a, "arrays"_a)
      .def("reset", &rsb::TaskKernel::Reset, "slots"_a.noconvert())
      .def("mark_terminated", &rsb::TaskKernel::MarkTerminated, "slots"_a.noconvert())
      .def("first_termination", &rsb::TaskKernel::FirstTermination, "rows"_a.noconvert(),
           "history"_a.noconvert(), "qpos_offset"_a, "out"_a.noconvert())
      .def("control", &rsb::TaskKernel::Control, "slots"_a.noconvert(), "reward"_a.noconvert(),
           "components"_a.noconvert())
      .def("metrics", &rsb::TaskKernel::Metrics, "slots"_a.noconvert(), "out"_a.noconvert())
      .def_prop_ro("name", &rsb::TaskKernel::name)
      .def_prop_ro("version", &rsb::TaskKernel::version)
      .def_prop_ro("num_components", &rsb::TaskKernel::num_components)
      .def_prop_ro("num_commands", &rsb::TaskKernel::num_commands)
      .def_prop_ro("num_state", &rsb::TaskKernel::num_state);
}
