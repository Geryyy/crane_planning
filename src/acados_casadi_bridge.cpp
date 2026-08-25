#include "acados_casadi_bridge.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <casadi/casadi.hpp>

extern "C" {
#include "acados/utils/external_function_generic.h"
#include "acados_c/external_function_interface.h"
}

namespace crane_planning
{
namespace
{

/// What one trampoline slot forwards into.
struct Slot
{
  bool taken{false};
  casadi::Function function{};

  /// The sparsity patterns in acados' `int` flavour, kept alive for the C side.
  std::vector<std::vector<int>> sparsity_in{};
  std::vector<std::vector<int>> sparsity_out{};

  int n_in{};
  int n_out{};
  int sz_arg{};
  int sz_res{};
  int sz_iw{};
  int sz_w{};
};

std::mutex & slot_mutex()
{
  static std::mutex mutex;
  return mutex;
}

std::array<Slot, kAcadosTrampolineSlots> & slots()
{
  static std::array<Slot, kAcadosTrampolineSlots> table;
  return table;
}

/// The CCS descriptor acados reads, in the shape its `casadi_nnz` expects.
/**
 * `{nrow, ncol, 1}` is the dense short form -- acados tests element two for
 * "dense" and multiplies out the shape. Everything bound here is densified
 * first, so the long form is never reached; it is written out anyway because a
 * silently wrong descriptor is read as garbage rather than refused.
 */
std::vector<int> compress_sparsity(const casadi::Sparsity & sparsity)
{
  const int rows = static_cast<int>(sparsity.size1());
  const int columns = static_cast<int>(sparsity.size2());
  if (sparsity.is_dense()) {
    return {rows, columns, 1};
  }
  std::vector<int> compressed;
  const std::vector<casadi_int> colind = sparsity.get_colind();
  const std::vector<casadi_int> row = sparsity.get_row();
  compressed.reserve(2U + colind.size() + row.size());
  compressed.push_back(rows);
  compressed.push_back(columns);
  for (const casadi_int value : colind) {
    compressed.push_back(static_cast<int>(value));
  }
  for (const casadi_int value : row) {
    compressed.push_back(static_cast<int>(value));
  }
  return compressed;
}

// The trampolines. One instantiation per slot, because acados takes a bare C
// function pointer and there is nowhere to put a `this`.

/// Forward one acados call into the live `casadi::Function` bound to this slot.
/**
 * **acados' own buffers may not be handed to CasADi, and neither may acados'
 * workspaces.** `casadi::Function::sz_arg()` and `sz_res()` are documented as
 * the *required length of the arg and res fields*, not as `n_in` and `n_out`:
 * CasADi treats those two pointer tables as scratch it owns for the duration of
 * a call. acados' `args` table is not scratch -- entry `idx_in_p` is the stage's
 * parameter pointer, set once at registration and expected to survive every
 * evaluation. Handing it over is what the first version of this bridge did, and
 * the failure was quiet rather than loud: the first evaluation on a horizon was
 * right, every later one read a clobbered parameter table, and the OCP came back
 * `ACADOS_NAN_DETECTED` at iteration zero with finite functions everywhere a
 * direct CasADi call was tried.
 *
 * The vector overload is CasADi's own answer to exactly this: it copies both
 * pointer tables, sizes `iw` and `w` itself from `sz_iw()`/`sz_w()`, and takes a
 * scoped memory checkout rather than assuming memory zero exists. Four small
 * allocations per evaluation is the price, and this is off the control cycle by
 * contract 10 -- the graph itself allocates on every call.
 */
template<int Index>
int slot_evaluate(const double ** arg, double ** res, int * /*iw*/, double * /*w*/, void * /*mem*/)
{
  const Slot & slot = slots()[static_cast<std::size_t>(Index)];
  slot.function(
    std::vector<const double *>(arg, arg + slot.sz_arg),
    std::vector<double *>(res, res + slot.sz_res));
  return 0;
}

template<int Index>
int slot_work(int * sz_arg, int * sz_res, int * sz_iw, int * sz_w)
{
  const Slot & slot = slots()[static_cast<std::size_t>(Index)];
  *sz_arg = slot.sz_arg;
  *sz_res = slot.sz_res;
  // acados allocates these and hands them to `slot_evaluate`, which does not use
  // them -- see the note there. They are reported truthfully all the same, so
  // that an acados which one day touches them finds room and not a short buffer.
  *sz_iw = slot.sz_iw;
  *sz_w = slot.sz_w;
  return 0;
}

template<int Index>
const int * slot_sparsity_in(int index)
{
  const Slot & slot = slots()[static_cast<std::size_t>(Index)];
  if (index < 0 || index >= static_cast<int>(slot.sparsity_in.size())) {
    return nullptr;
  }
  return slot.sparsity_in[static_cast<std::size_t>(index)].data();
}

template<int Index>
const int * slot_sparsity_out(int index)
{
  const Slot & slot = slots()[static_cast<std::size_t>(Index)];
  if (index < 0 || index >= static_cast<int>(slot.sparsity_out.size())) {
    return nullptr;
  }
  return slot.sparsity_out[static_cast<std::size_t>(index)].data();
}

template<int Index>
int slot_n_in()
{
  return slots()[static_cast<std::size_t>(Index)].n_in;
}

template<int Index>
int slot_n_out()
{
  return slots()[static_cast<std::size_t>(Index)].n_out;
}

/// The six pointers acados wants, for one slot.
struct Trampoline
{
  int (* evaluate)(const double **, double **, int *, double *, void *);
  int (* work)(int *, int *, int *, int *);
  const int * (* sparsity_in)(int);
  const int * (* sparsity_out)(int);
  int (* n_in)();
  int (* n_out)();
};

template<int Index>
constexpr Trampoline make_trampoline()
{
  return Trampoline{
    &slot_evaluate<Index>, &slot_work<Index>, &slot_sparsity_in<Index>,
    &slot_sparsity_out<Index>, &slot_n_in<Index>, &slot_n_out<Index>};
}

template<int ... Index>
constexpr std::array<Trampoline, sizeof...(Index)> make_trampolines(
  std::integer_sequence<int, Index...>)
{
  return {{make_trampoline<Index>() ...}};
}

const std::array<Trampoline, kAcadosTrampolineSlots> & trampolines()
{
  static const std::array<Trampoline, kAcadosTrampolineSlots> table =
    make_trampolines(std::make_integer_sequence<int, kAcadosTrampolineSlots>{});
  return table;
}

}  // namespace

void * AcadosCasadiFunction::handle(std::size_t instance) const noexcept
{
  return instance < handles_.size() ? handles_[instance] : nullptr;
}

AcadosCasadiFunction::~AcadosCasadiFunction()
{
  for (void * raw : handles_) {
    auto * fun = static_cast<external_function_external_param_casadi *>(raw);
    external_function_external_param_casadi_free(fun);
    delete fun;
  }
  handles_.clear();
  if (slot_ >= 0) {
    const std::lock_guard<std::mutex> guard(slot_mutex());
    Slot & slot = slots()[static_cast<std::size_t>(slot_)];
    slot.function = casadi::Function{};
    slot.sparsity_in.clear();
    slot.sparsity_out.clear();
    slot.taken = false;
    slot_ = -1;
  }
}

std::string AcadosCasadiFunction::bind(const casadi::Function & function, std::size_t instances)
{
  if (!handles_.empty()) {
    return "this AcadosCasadiFunction is already bound";
  }
  if (instances == 0U) {
    return "a bound function needs at least one acados instance";
  }
  if (function.is_null()) {
    return "cannot bind a null casadi::Function";
  }
  if (function.n_in() < 1) {
    return "a bound function takes its acados parameter vector as its last "
           "input, so it needs at least one input";
  }

  int index = -1;
  {
    const std::lock_guard<std::mutex> guard(slot_mutex());
    for (int candidate = 0; candidate < kAcadosTrampolineSlots; ++candidate) {
      if (!slots()[static_cast<std::size_t>(candidate)].taken) {
        index = candidate;
        slots()[static_cast<std::size_t>(candidate)].taken = true;
        break;
      }
    }
  }
  if (index < 0) {
    return "no acados trampoline slot is free; more solvers are alive at once "
           "than the fixed table allows";
  }

  Slot & slot = slots()[static_cast<std::size_t>(index)];
  slot.function = function;
  slot.n_in = static_cast<int>(function.n_in());
  slot.n_out = static_cast<int>(function.n_out());
  slot.sz_arg = static_cast<int>(function.sz_arg());
  slot.sz_res = static_cast<int>(function.sz_res());
  slot.sz_iw = static_cast<int>(function.sz_iw());
  slot.sz_w = static_cast<int>(function.sz_w());
  slot.sparsity_in.reserve(static_cast<std::size_t>(function.n_in()));
  for (casadi_int input = 0; input < function.n_in(); ++input) {
    slot.sparsity_in.push_back(compress_sparsity(function.sparsity_in(input)));
  }
  slot.sparsity_out.reserve(static_cast<std::size_t>(function.n_out()));
  for (casadi_int output = 0; output < function.n_out(); ++output) {
    slot.sparsity_out.push_back(compress_sparsity(function.sparsity_out(output)));
  }

  const Trampoline & trampoline = trampolines()[static_cast<std::size_t>(index)];
  slot_ = index;
  handles_.reserve(instances);
  for (std::size_t instance = 0; instance < instances; ++instance) {
    auto * fun = new external_function_external_param_casadi{};
    fun->casadi_fun = trampoline.evaluate;
    fun->casadi_work = trampoline.work;
    fun->casadi_sparsity_in = trampoline.sparsity_in;
    fun->casadi_sparsity_out = trampoline.sparsity_out;
    fun->casadi_n_in = trampoline.n_in;
    fun->casadi_n_out = trampoline.n_out;

    external_function_opts opts;
    external_function_opts_set_to_default(&opts);
    external_function_external_param_casadi_create(fun, &opts);
    handles_.push_back(fun);
  }
  return {};
}

}  // namespace crane_planning
