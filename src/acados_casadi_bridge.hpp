#ifndef ACADOS_CASADI_BRIDGE_HPP_
#define ACADOS_CASADI_BRIDGE_HPP_

#include <cstddef>
#include <string>
#include <vector>

#include <casadi/casadi.hpp>

// The seam between `crane_model`'s CasADi graph and acados' C core.
//
// acados takes a nonlinear function as an `external_function_external_param_casadi`:
// six **plain C function pointers** following the calling convention of the C
// code `casadi::CodeGenerator` emits. `crane_model::symbolic::CasadiGraph` hands
// over `casadi::Function` *objects* instead, so something has to turn one into
// the other. Two routes exist and only one of them is honest here:
//
//   * generate C, compile it and `dlopen` it at run time -- which would make the
//     planner node depend on a compiler being installed on the machine it runs
//     on, and would put a `cc` invocation inside a service call;
//   * hand acados C trampolines that forward into the live `casadi::Function`.
//
// This is the second. `AcadosCasadiFunction` binds one `casadi::Function` to one
// slot of a fixed table of trampolines and fills in the acados struct. Nothing is
// generated, nothing is compiled and nothing is written to disk.
//
// Three conventions are load-bearing and none is documented anywhere but in
// acados' own source:
//
//   * **`p` is the last input.** `external_function_external_param_casadi`
//     computes `idx_in_p = n_in - 1` and writes the stage's parameter pointer
//     there (`acados/utils/external_function_generic.c`). A bound function must
//     therefore take its parameter vector as its final argument, always, even
//     when it does not use it.
//   * **acados' `casadi_int` is `int`.** The struct declares
//     `int (*casadi_fun)(const double**, double**, int*, double*, void*)`, while
//     this image's CasADi is built with a 64-bit `casadi_int`. The trampoline
//     therefore ignores acados' integer workspace and uses its own, which is the
//     only place the two integer widths could have met.
//   * **`sz_arg` may exceed `n_in`.** acados walks the argument list up to
//     `sz_arg` and asks for a sparsity at every index; generated CasADi code
//     answers `NULL` past the last real input and acados reads that as an empty
//     argument, so the trampolines do the same.
//
// Everything here is off the control cycle: binding allocates, and evaluation
// goes through `casadi::Function::operator()`, which is not real time. That is
// the same standing contract §10 gives the graph itself.

namespace crane_planning
{

/// One `casadi::Function`, callable by acados at every stage of a horizon.
/**
 * acados wants the parameter vector to arrive by *pointer*, and it takes that
 * pointer from `nlp_in->parameter_values[stage]` when the function is registered
 * -- so a horizon whose stages carry different parameters needs one acados
 * struct per stage. The **trampolines** need no such thing: they only expose the
 * CasADi function, which is the same at every stage. One bind therefore takes
 * one trampoline slot and hands out as many structs as there are stages.
 *
 * Getting that the wrong way round is expensive and quiet: a slot per stage
 * exhausts the table on the first horizon of any length, and `handle()` then
 * returns null into a C setter that dereferences it.
 *
 * The object owns the structs and the slot; destroying it frees both. It is
 * neither copyable nor movable, because acados stores the *addresses* of the
 * structs and a move would leave them dangling -- hold it by `std::unique_ptr`,
 * or in a container that does not reallocate.
 *
 * Not thread safe. One bound function carries one integer and one float
 * workspace, so two threads evaluating it would share them. A solver is driven
 * from one thread; that is the contract.
 */
class AcadosCasadiFunction
{
public:
  AcadosCasadiFunction() = default;
  ~AcadosCasadiFunction();
  AcadosCasadiFunction(const AcadosCasadiFunction &) = delete;
  AcadosCasadiFunction & operator=(const AcadosCasadiFunction &) = delete;
  AcadosCasadiFunction(AcadosCasadiFunction &&) = delete;
  AcadosCasadiFunction & operator=(AcadosCasadiFunction &&) = delete;

  /// Bind `function` as `instances` acados structs sharing one trampoline slot.
  /**
   * `function`'s last input must be the acados parameter vector. Returns an
   * empty string on success and the reason on failure; the only failure that is
   * not a programming error is running out of trampoline slots, which happens
   * when more distinct functions are alive at once than
   * `kAcadosTrampolineSlots` allows.
   */
  [[nodiscard]] std::string bind(const casadi::Function & function, std::size_t instances);

  /// The struct to hand `ocp_nlp_*_model_set_external_param_fun`. Null until bound.
  /**
   * Typed `void *` on purpose: acados declares the struct as an anonymous
   * `typedef`, so there is no tag to forward declare and a header that named the
   * type would have to pull all of acados into every consumer. The setters take
   * `void *` anyway.
   */
  [[nodiscard]] void * handle(std::size_t instance) const noexcept;

private:
  std::vector<void *> handles_{};
  int slot_{-1};
};

/// How many distinct `casadi::Function`s may be bound at once, across all solvers.
/**
 * A trampoline is a C function pointer with no context argument, so each bound
 * function needs its own instantiation and the number of them is fixed at
 * compile time. One timing OCP binds seven -- not seven per stage, seven in
 * total. The table is sized for several solvers alive at once rather than for
 * one, because a node asked to re-plan while a plan is in flight would otherwise
 * refuse for a reason that has nothing to do with the machine.
 */
inline constexpr int kAcadosTrampolineSlots = 64;

}  // namespace crane_planning

#endif  // ACADOS_CASADI_BRIDGE_HPP_
