#include "crane_planning/sampling_planner.hpp"

#include <ompl/base/MotionValidator.h>
#include <ompl/base/PlannerStatus.h>
#include <ompl/base/PlannerTerminationCondition.h>
#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/StateSampler.h>
#include <ompl/base/StateSpace.h>
#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/datastructures/NearestNeighborsLinear.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/util/Console.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace crane_planning
{
namespace
{

namespace ob = ompl::base;
namespace og = ompl::geometric;

using crane_model::ErrorCode;
using crane_model::Q;
using crane_model::Result;
using crane_model::Status;

/// Longest a single motion may be subdivided into, at the stated resolution.
/**
 * A motion the resolution asks for more pieces than this is refused rather than
 * validated coarsely, which is the same rule `check_path` applies to a whole
 * path with `max_samples`. Nothing the search itself proposes comes near it --
 * `extension_span` bounds a tree extension -- so what this really bounds is a
 * shortcut chord across most of the path.
 */
constexpr double kMaxSubdivisions = 4096.0;

Status failure(ErrorCode code, std::string message)
{
  return Status{code, std::move(message)};
}

/// A refusal that says which of 4.4's two mechanisms it came from.
Status refuse(ErrorCode code, const std::string & why)
{
  return failure(
    code,
    "the sampling fallback of trajectory_planning 4.4 could not produce a path: " + why +
    ". The structured primitive was generated and checked first and was not accepted, which is "
    "the only reason the fallback ran at all");
}

double coordinate(const ob::State * state, std::size_t row)
{
  return state->as<ob::CompoundState>()
         ->as<ob::RealVectorStateSpace::StateType>(static_cast<unsigned int>(row))->values[0];
}

void set_coordinate(ob::State * state, std::size_t row, double value)
{
  state->as<ob::CompoundState>()
  ->as<ob::RealVectorStateSpace::StateType>(static_cast<unsigned int>(row))->values[0] = value;
}

PathVector vector_of(const ob::State * state)
{
  PathVector q_a = PathVector::Zero();
  for (std::size_t row = 0; row < kPathDof; ++row) {
    q_a[static_cast<Eigen::Index>(row)] = coordinate(state, row);
  }
  return q_a;
}

void write_state(const PathVector & q_a, ob::State * state)
{
  for (std::size_t row = 0; row < kPathDof; ++row) {
    set_coordinate(state, row, q_a[static_cast<Eigen::Index>(row)]);
  }
}

/// The path-space part of a canonical eight-vector.
PathVector path_of(const Q & q)
{
  PathVector q_a = PathVector::Zero();
  for (std::size_t row = 0; row < kPathDof; ++row) {
    q_a[static_cast<Eigen::Index>(row)] = q[static_cast<Eigen::Index>(kActuatedRows[row])];
  }
  return q_a;
}

/// The canonical eight of one path vector, tool held, passive pair left at zero.
Q configuration_of(const PathVector & q_a, double q8)
{
  Q q = Q::Zero();
  for (std::size_t row = 0; row < kPathDof; ++row) {
    q[static_cast<Eigen::Index>(kActuatedRows[row])] = q_a[static_cast<Eigen::Index>(row)];
  }
  q[static_cast<Eigen::Index>(kActuatedRows[kToolRow])] = q8;
  return q;
}

/// The actuated six of one path vector, tool held.
crane_model::QA actuated_of(const PathVector & q_a, double q8)
{
  crane_model::QA actuated = crane_model::QA::Zero();
  for (std::size_t row = 0; row < kPathDof; ++row) {
    actuated[static_cast<Eigen::Index>(row)] = q_a[static_cast<Eigen::Index>(row)];
  }
  actuated[static_cast<Eigen::Index>(kToolRow)] = q8;
  return actuated;
}

/// A uniform sampler over the bounds, drawing from a seed and from nothing else.
/**
 * OMPL's own generator is seeded once per process and complains when it is seeded
 * again, so a suite whose second test wants a fixed seed cannot have one from it.
 * This sampler carries its own `std::mt19937`, which is what makes a seeded run
 * reproducible wherever in a process it happens to run.
 */
class SeededSampler : public ob::StateSampler
{
public:
  SeededSampler(const ob::StateSpace * space, StateSpaceBounds bounds, std::uint32_t seed)
  : ob::StateSampler(space), bounds_(std::move(bounds)), generator_(seed) {}

  void sampleUniform(ob::State * state) override
  {
    for (std::size_t row = 0; row < kPathDof; ++row) {
      const Eigen::Index axis = static_cast<Eigen::Index>(row);
      std::uniform_real_distribution<double> uniform(bounds_.lower[axis], bounds_.upper[axis]);
      set_coordinate(state, row, uniform(generator_));
    }
  }

  void sampleUniformNear(ob::State * state, const ob::State * near, double distance) override
  {
    for (std::size_t row = 0; row < kPathDof; ++row) {
      const Eigen::Index axis = static_cast<Eigen::Index>(row);
      const double centre = coordinate(near, row);
      const double low = std::max(bounds_.lower[axis], centre - distance);
      const double high = std::max(low, std::min(bounds_.upper[axis], centre + distance));
      std::uniform_real_distribution<double> uniform(low, high);
      set_coordinate(state, row, uniform(generator_));
    }
  }

  void sampleGaussian(ob::State * state, const ob::State * mean, double stdDev) override
  {
    for (std::size_t row = 0; row < kPathDof; ++row) {
      const Eigen::Index axis = static_cast<Eigen::Index>(row);
      std::normal_distribution<double> normal(coordinate(mean, row), stdDev);
      const double drawn = normal(generator_);
      set_coordinate(
        state, row, std::min(bounds_.upper[axis], std::max(bounds_.lower[axis], drawn)));
    }
  }

private:
  StateSpaceBounds bounds_;
  std::mt19937 generator_;
};

/// The state validity checker: issue 041's collision check, envelope and all.
/**
 * Not a second collision model and not a cheaper one. The passive pair is solved
 * here with `Model::passive_equilibrium` exactly as `check_path` solves it per
 * sample, so a sampled configuration is judged at the pose the tool really hangs
 * in, over the sway envelope of 4.3, against the scene, the truck and the crane
 * itself.
 *
 * Two answers are deliberately different from each other. A configuration the
 * check ran on and refused is **invalid** and the search goes elsewhere. A check
 * that could not be *run* -- an unusable scene, a description with no collision
 * geometry -- is a `status_` that ends the search and is reported as itself,
 * because "no path was found" would read as "there is no way round" when what
 * happened is that nothing was ever checked.
 */
class ConfigurationValidity : public ob::StateValidityChecker
{
public:
  ConfigurationValidity(
    const ob::SpaceInformationPtr & si, const crane_model::Model & model,
    const crane_model::CollisionScene & scene, const crane_model::Payload & payload,
    const PayloadShape & shape, const CollisionSettings & settings, double q8, double step_m)
  : ob::StateValidityChecker(si), model_(model), scene_(scene), payload_(payload), shape_(shape),
    settings_(settings), q8_(q8), step_m_(step_m) {}

  bool isValid(const ob::State * state) const override
  {
    // Out of allowance, or a check that could not be run: stop answering rather
    // than answer "clear". The termination condition reads the same two and ends
    // the search, and the refusal names which of them it was.
    if (!status_.ok() || checks_ >= budget_) {
      return false;
    }
    if (!si_->satisfiesBounds(state)) {
      return false;
    }
    ++checks_;

    const PathVector q_a = vector_of(state);
    if (!q_a.allFinite()) {
      return false;
    }

    // q5 and q6 are dynamics outputs and never path variables (4.1), so they are
    // solved at every configuration and never interpolated.
    auto settled = model_.passive_equilibrium(actuated_of(q_a, q8_), payload_);
    if (!settled.ok()) {
      // A configuration whose passive subsystem has no steady state is not one to
      // plan through. It is not a failure of the check either -- the arm folded
      // past its working range has no hanging pose and is not a state this
      // machine is asked to pass through.
      return false;
    }
    Q q = configuration_of(q_a, q8_);
    q.segment<2>(4) = settled.value();

    auto checked = check_configuration(model_, scene_, shape_, settings_, q, step_m_);
    if (!checked.ok()) {
      status_ = checked.status();
      return false;
    }
    return checked.value().clear;
  }

  [[nodiscard]] std::size_t checks() const noexcept {return checks_;}
  [[nodiscard]] const Status & status() const noexcept {return status_;}
  [[nodiscard]] bool spent() const noexcept {return checks_ >= budget_;}

  /// Give the checker a fresh allowance of `more` configurations from here on.
  void allow(std::size_t more) noexcept {budget_ = checks_ + more;}

private:
  const crane_model::Model & model_;
  const crane_model::CollisionScene & scene_;
  const crane_model::Payload & payload_;
  const PayloadShape & shape_;
  const CollisionSettings & settings_;
  double q8_;
  double step_m_;
  mutable std::size_t checks_{0};
  std::size_t budget_{0};
  mutable Status status_{};
};

/// Motion validation at the resolution `check_path` samples at, not at endpoints.
/**
 * `collision.hpp`'s note is the rule and it applies to a straight line between
 * two sampled states exactly as it applies to a fitted path: a motion checked
 * only at its ends is not checked, and the step is measured as the largest
 * distance any watched frame moves. OMPL's own `DiscreteMotionValidator` would
 * have subdivided in state-space distance instead, which says nothing about how
 * far the geometry travelled -- half a radian of slewing moves the tool a great
 * deal further with the boom out than with it in.
 */
class TravelMotionValidator : public ob::MotionValidator
{
public:
  TravelMotionValidator(
    const ob::SpaceInformationPtr & si, const crane_model::Model & model, double q8, double step_m)
  : ob::MotionValidator(si), model_(model), q8_(q8), step_m_(step_m) {}

  bool checkMotion(const ob::State * s1, const ob::State * s2) const override
  {
    std::pair<ob::State *, double> ignored{nullptr, 0.0};
    return checkMotion(s1, s2, ignored);
  }

  bool checkMotion(
    const ob::State * s1, const ob::State * s2,
    std::pair<ob::State *, double> & lastValid) const override
  {
    const std::size_t pieces = subdivisions(s1, s2);
    if (pieces == 0U) {
      stop_at(s1, s2, 0.0, lastValid);
      ++invalid_;
      return false;
    }

    bool valid = true;
    ob::State * probe = si_->allocState();
    for (std::size_t piece = 1U; piece < pieces; ++piece) {
      const double when = static_cast<double>(piece) / static_cast<double>(pieces);
      si_->getStateSpace()->interpolate(s1, s2, when, probe);
      if (!si_->isValid(probe)) {
        stop_at(
          s1, s2, static_cast<double>(piece - 1U) / static_cast<double>(pieces), lastValid);
        valid = false;
        break;
      }
    }
    si_->freeState(probe);

    // `s1` is the caller's to have checked; `s2` is not, and a motion whose far
    // end is in something is not a valid motion however clear the inside was.
    if (valid && !si_->isValid(s2)) {
      stop_at(s1, s2, static_cast<double>(pieces - 1U) / static_cast<double>(pieces), lastValid);
      valid = false;
    }

    if (valid) {
      ++valid_;
    } else {
      ++invalid_;
    }
    return valid;
  }

  [[nodiscard]] const Status & status() const noexcept {return status_;}

private:
  /// How many pieces the stated resolution asks this motion to be cut into.
  /**
   * Zero means the travel could not be measured or the motion is longer than
   * `kMaxSubdivisions` pieces of it, and both are answered as an invalid motion.
   */
  std::size_t subdivisions(const ob::State * s1, const ob::State * s2) const
  {
    auto travel = watched_frame_travel_m(
      model_, configuration_of(vector_of(s1), q8_), configuration_of(vector_of(s2), q8_));
    if (!travel.ok()) {
      status_ = travel.status();
      return 0U;
    }
    const double pieces = std::ceil(travel.value() / step_m_);
    if (!std::isfinite(pieces) || pieces > kMaxSubdivisions) {
      return 0U;
    }
    return static_cast<std::size_t>(std::max(1.0, pieces));
  }

  void stop_at(
    const ob::State * s1, const ob::State * s2, double when,
    std::pair<ob::State *, double> & lastValid) const
  {
    lastValid.second = when;
    if (lastValid.first != nullptr) {
      si_->getStateSpace()->interpolate(s1, s2, when, lastValid.first);
    }
  }

  const crane_model::Model & model_;
  double q8_;
  double step_m_;
  mutable Status status_{};
};

std::string seconds(double value)
{
  return std::to_string(value) + " s";
}

}  // namespace

const char * mechanism_name(PathMechanism mechanism) noexcept
{
  switch (mechanism) {
    case PathMechanism::StructuredPrimitive:
      return "the structured lift/traverse/descend primitive of trajectory_planning 4.4";
    case PathMechanism::SamplingFallback:
      return "the RRT-Connect sampling fallback of trajectory_planning 4.4, smoothed and "
             "re-checked per 4.5";
  }
  return "an unknown mechanism";
}

crane_model::Result<StateSpaceBounds> state_space_bounds(
  const JointLimits & limits, const PathVector & q_a_start, const PathVector & q_a_goal,
  const SamplingSettings & settings)
{
  if (!q_a_start.allFinite() || !q_a_goal.allFinite()) {
    return Result<StateSpaceBounds>::failure(
      failure(ErrorCode::NonFiniteInput, "an endpoint of the sampled search is not finite"));
  }
  if (!(settings.unbounded_margin_rad > 0.0) || !std::isfinite(settings.unbounded_margin_rad)) {
    return Result<StateSpaceBounds>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "an unbounded axis needs a positive finite margin to be searched over at all"));
  }

  StateSpaceBounds bounds;
  for (std::size_t row = 0; row < kPathDof; ++row) {
    const Eigen::Index axis = static_cast<Eigen::Index>(row);
    const AxisLimit & limit = limits.axis[row];
    if (!(limit.dq_max > 0.0) || !std::isfinite(limit.dq_max)) {
      return Result<StateSpaceBounds>::failure(
        failure(
          ErrorCode::InvalidArgument,
          "path coordinate " + std::to_string(row) +
          " has no positive velocity limit, so the state space has no unit to measure a distance "
          "in and the search would be comparing radians with metres"));
    }
    bounds.weight[axis] = 1.0 / limit.dq_max;
    bounds.bounded_by_description[row] = limit.bounded;

    if (limit.bounded) {
      bounds.lower[axis] = limit.lower;
      bounds.upper[axis] = limit.upper;
      const double start = q_a_start[axis];
      const double goal = q_a_goal[axis];
      if (start < limit.lower || start > limit.upper || goal < limit.lower || goal > limit.upper) {
        return Result<StateSpaceBounds>::failure(
          failure(
            ErrorCode::InvalidArgument,
            "path coordinate " + std::to_string(row) + " runs from " + std::to_string(start) +
            " to " + std::to_string(goal) + ", outside the [" + std::to_string(limit.lower) +
            ", " + std::to_string(limit.upper) +
            "] the description gives it, so there is no state space to sample that contains both "
            "endpoints"));
      }
      continue;
    }

    // A `continuous` joint -- the rotator on both machines. It has no range in
    // the description, which is a statement and not an omission, and a sampler
    // still needs one. Half a turn either side of what the endpoints ask for is
    // every distinct pose it has, because q7 and q7 + 2 pi are the same geometry.
    bounds.lower[axis] = std::min(q_a_start[axis], q_a_goal[axis]) - settings.unbounded_margin_rad;
    bounds.upper[axis] = std::max(q_a_start[axis], q_a_goal[axis]) + settings.unbounded_margin_rad;
  }
  return Result<StateSpaceBounds>::success(bounds);
}

std::vector<PathVector> shortcut(
  const std::vector<PathVector> & waypoints, std::size_t attempts, std::uint32_t seed,
  const std::function<bool(const PathVector &, const PathVector &)> & accept)
{
  std::vector<PathVector> kept = waypoints;
  if (kept.size() < 3U || attempts == 0U || !accept) {
    return kept;
  }

  std::mt19937 generator(seed);
  for (std::size_t attempt = 0; attempt < attempts && kept.size() > 2U; ++attempt) {
    std::uniform_int_distribution<std::size_t> pick(0U, kept.size() - 1U);
    std::size_t first = pick(generator);
    std::size_t second = pick(generator);
    if (first > second) {
      std::swap(first, second);
    }
    // Neighbours have nothing between them to drop, so there is no shortcut to
    // take and no reason to pay for a motion check to find that out.
    if (second < first + 2U) {
      continue;
    }
    if (!accept(kept[first], kept[second])) {
      continue;
    }
    kept.erase(
      std::next(kept.begin(), static_cast<std::ptrdiff_t>(first + 1U)),
      std::next(kept.begin(), static_cast<std::ptrdiff_t>(second)));
  }
  return kept;
}

std::string describe(const SamplingPlan & plan)
{
  std::string text =
    "the structured primitive of trajectory_planning 4.4 was refused, so the sampling fallback "
    "of that section answered: RRT-Connect over the five path coordinates of 4.1 -- never the "
    "passive pair -- returned " + std::to_string(plan.search_states) +
    " waypoints after " + std::to_string(plan.validity_checks) + " configurations checked in " +
    seconds(plan.search_s) + " from seed " + std::to_string(plan.seed) +
    ". 4.5's mandatory smoothing then ran in full: the shortcut pass left " +
    std::to_string(plan.smoothed_states) + " of those waypoints at a further " +
    std::to_string(plan.shortcut_checks) +
    " configurations, the C2 fit turned them into " +
    std::to_string(plan.path.segment_count()) +
    " segments continuous in q_a' and q_a'', and the whole curve was then re-checked because "
    "smoothing moves a path off the one that was cleared. " + describe(plan.recheck);
  return text;
}

crane_model::Result<SamplingPlan> plan_sampled_path(
  const crane_model::Model & model, const JointLimits & limits, const PathFitSettings & fit,
  const CollisionSettings & collision, const SamplingSettings & settings,
  const SamplingRequest & request)
{
  if (request.collision_scene == nullptr) {
    return Result<SamplingPlan>::failure(
      refuse(
        ErrorCode::NotReady,
        "no scene has been received on /crane/collision_scene, and there is nothing to sample a "
        "way round without one"));
  }
  if (!request.q_start.allFinite() || !request.q_goal.allFinite()) {
    return Result<SamplingPlan>::failure(
      refuse(ErrorCode::NonFiniteInput, "an endpoint configuration is not finite"));
  }
  if (!(settings.time_budget_s > 0.0) || !std::isfinite(settings.time_budget_s) ||
    settings.max_validity_checks < 2U || !(settings.extension_span >= 0.0) ||
    !std::isfinite(settings.extension_span))
  {
    return Result<SamplingPlan>::failure(
      refuse(
        ErrorCode::InvalidArgument,
        "a search needs a positive wall-clock budget, an allowance of at least two configurations "
        "and a non-negative extension span"));
  }

  const double q8_start = request.q_start[static_cast<Eigen::Index>(kActuatedRows[kToolRow])];
  const double q8_goal = request.q_goal[static_cast<Eigen::Index>(kActuatedRows[kToolRow])];
  if (std::abs(q8_goal - q8_start) > 1.0e-9) {
    return Result<SamplingPlan>::failure(
      refuse(
        ErrorCode::InvalidArgument,
        "the tool coordinate is asked to move from " + std::to_string(q8_start) + " to " +
        std::to_string(q8_goal) +
        ". 4.1 says q8 is not a path variable, so this search holds it; what opens and closes the "
        "gripper is /crane/plan_grip, which is issue 044"));
  }

  const PathVector start = path_of(request.q_start);
  const PathVector goal = path_of(request.q_goal);

  auto bounds = state_space_bounds(limits, start, goal, settings);
  if (!bounds.ok()) {
    return Result<SamplingPlan>::failure(refuse(bounds.status().code, bounds.status().message));
  }
  // The step, taken from the same place `check_path` takes it, so the search and
  // the re-check resolve the scene identically.
  auto resolution = resolve_scene(*request.collision_scene, request.payload_shape, collision);
  if (!resolution.ok()) {
    return Result<SamplingPlan>::failure(resolution.status());
  }
  const double step_m = resolution.value().step_m;

  // A planner inside a service call does not talk to stdout.
  ompl::msg::setLogLevel(ompl::msg::LOG_WARN);

  // Five subspaces rather than one five-vector, so that the distance the search
  // reasons about is a sum of times at each axis's own limit instead of a norm
  // over a vector that mixes radians with metres.
  auto space = std::make_shared<ob::CompoundStateSpace>();
  for (std::size_t row = 0; row < kPathDof; ++row) {
    const Eigen::Index axis = static_cast<Eigen::Index>(row);
    auto coordinate_space = std::make_shared<ob::RealVectorStateSpace>(1U);
    ob::RealVectorBounds range(1);
    range.setLow(0, bounds.value().lower[axis]);
    range.setHigh(0, bounds.value().upper[axis]);
    coordinate_space->setBounds(range);
    space->addSubspace(coordinate_space, bounds.value().weight[axis]);
  }
  space->lock();
  space->setStateSamplerAllocator(
    [drawn = bounds.value(), seed = settings.seed](const ob::StateSpace * space_of) {
      return std::make_shared<SeededSampler>(space_of, drawn, seed);
    });

  auto space_information = std::make_shared<ob::SpaceInformation>(space);
  auto validity = std::make_shared<ConfigurationValidity>(
    space_information, model, *request.collision_scene, request.payload, request.payload_shape,
    collision, q8_start, step_m);
  // Two, for the two endpoints, before the search's own allowance is opened.
  // `max_validity_checks` is a cap on what the *search* may look at, and a
  // deployment that set it to two would otherwise be told its budget ran out when
  // what happened is that the endpoints used it up.
  validity->allow(2U);
  auto motion = std::make_shared<TravelMotionValidator>(
    space_information, model, q8_start, step_m);
  space_information->setStateValidityChecker(validity);
  space_information->setMotionValidator(motion);
  space_information->setup();

  ob::ScopedState<> start_state(space);
  ob::ScopedState<> goal_state(space);
  write_state(start, start_state.get());
  write_state(goal, goal_state.get());

  // The two endpoints, before the search. A start or a goal the check already
  // refuses is a different answer from an exhausted budget: "the machine is
  // already in something" and "there is no way round in the time given" are not
  // the same thing to tell an operator.
  const auto endpoint_refusal = [](const char * which) {
      return refuse(
        ErrorCode::InvalidArgument,
        std::string("the ") + which +
        " configuration is itself refused by the collision check of trajectory_planning 4.2 and "
        "4.3, so no path from it can be cleared and no budget would have helped");
    };
  if (!validity->isValid(start_state.get())) {
    if (!validity->status().ok()) {
      return Result<SamplingPlan>::failure(validity->status());
    }
    return Result<SamplingPlan>::failure(endpoint_refusal("start"));
  }
  if (!validity->isValid(goal_state.get())) {
    if (!validity->status().ok()) {
      return Result<SamplingPlan>::failure(validity->status());
    }
    return Result<SamplingPlan>::failure(endpoint_refusal("goal"));
  }

  const std::size_t endpoint_checks = validity->checks();
  validity->allow(settings.max_validity_checks);

  auto problem = std::make_shared<ob::ProblemDefinition>(space_information);
  problem->setStartAndGoalStates(start_state, goal_state);

  auto planner = std::make_shared<og::RRTConnect>(space_information);
  planner->setProblemDefinition(problem);
  if (settings.extension_span > 0.0) {
    planner->setRange(settings.extension_span);
  }
  // Exact, and with no random pivots in it. The tree is small because every node
  // costs a `Model::passive_equilibrium` solve, so a linear nearest-neighbour
  // search is free beside one check -- and OMPL's default GNAT picks its pivots
  // from the global generator, which is the last thing in the run that would
  // otherwise make a seeded search depend on what ran before it.
  //
  // This sets the planner up on its way through, which is why there is no
  // `setup()` call after it: a second one is a warning on stderr and nothing else.
  planner->setNearestNeighbors<ompl::NearestNeighborsLinear>();

  const auto began = std::chrono::steady_clock::now();
  const auto elapsed = [&began]() {
      return std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
    };
  const ob::PlannerTerminationCondition budget(
    [&]() {
      return !validity->status().ok() || !motion->status().ok() || validity->spent() ||
             elapsed() >= settings.time_budget_s;
    });
  const ob::PlannerStatus outcome = planner->solve(budget);

  SamplingPlan plan;
  plan.seed = settings.seed;
  plan.search_s = elapsed();
  plan.validity_checks = validity->checks() - endpoint_checks;

  if (!validity->status().ok()) {
    return Result<SamplingPlan>::failure(validity->status());
  }
  if (!motion->status().ok()) {
    return Result<SamplingPlan>::failure(motion->status());
  }
  if (outcome != ob::PlannerStatus::EXACT_SOLUTION) {
    // An approximate solution is refused with the rest. A path that stops short
    // of the goal is not a worse answer than none; it is an answer that reads as
    // success, and stage 2 would time the machine straight into whatever the
    // search had not got round yet.
    return Result<SamplingPlan>::failure(
      refuse(
        ErrorCode::NotReady,
        std::string("RRT-Connect came back with '") + outcome.asString() + "' after " +
        seconds(plan.search_s) + " of a " + seconds(settings.time_budget_s) +
        " wall-clock budget and " + std::to_string(plan.validity_checks) + " of the " +
        std::to_string(settings.max_validity_checks) +
        " configurations it was allowed to check. Both caps are per call and are what keep an "
        "unbounded stochastic search out of a service call; the replanning loop's own latency "
        "bound is issue 045's and is not this one"));
  }

  auto solution = std::static_pointer_cast<og::PathGeometric>(problem->getSolutionPath());
  if (!solution || solution->getStateCount() < 2U) {
    return Result<SamplingPlan>::failure(
      refuse(
        ErrorCode::NotReady,
        "the search reported a solution and produced no path with two ends to it"));
  }
  std::vector<PathVector> waypoints;
  waypoints.reserve(solution->getStateCount());
  for (std::size_t index = 0; index < solution->getStateCount(); ++index) {
    waypoints.push_back(vector_of(solution->getState(index)));
  }
  plan.search_states = waypoints.size();

  // 4.5, step one. The pass is given a fresh allowance rather than the search's
  // leftovers, so how hard a search was does not decide how much smoothing the
  // path it found is allowed -- and running out here costs shortcuts, never
  // correctness, because every replacement is validated before it is taken.
  const std::size_t after_search = validity->checks();
  validity->allow(settings.max_validity_checks);
  ob::ScopedState<> from(space);
  ob::ScopedState<> to(space);
  const auto accept = [&](const PathVector & first, const PathVector & second) {
      write_state(first, from.get());
      write_state(second, to.get());
      return space_information->checkMotion(from.get(), to.get());
    };
  const std::vector<PathVector> smoothed =
    shortcut(waypoints, settings.shortcut_attempts, settings.seed + 1U, accept);
  plan.smoothed_states = smoothed.size();
  plan.shortcut_checks = validity->checks() - after_search;
  if (!validity->status().ok()) {
    return Result<SamplingPlan>::failure(validity->status());
  }
  if (!motion->status().ok()) {
    return Result<SamplingPlan>::failure(motion->status());
  }

  // 4.5, step two: the same C2 fit the primitive is built with, so both producers
  // hand stage 2 a path with the same two derivatives defined everywhere.
  PathFitRequest fitted;
  fitted.waypoints = smoothed;
  fitted.segment_names.reserve(smoothed.size() - 1U);
  for (std::size_t segment = 1U; segment < smoothed.size(); ++segment) {
    fitted.segment_names.push_back("sampled segment " + std::to_string(segment));
  }
  fitted.q8_start = q8_start;
  fitted.q8_goal = q8_goal;
  auto path = fit_c2_path(fitted, limits, fit);
  if (!path.ok()) {
    return Result<SamplingPlan>::failure(
      refuse(
        path.status().code,
        "the C2 fit that 4.5 makes mandatory refused the sampled polyline: " +
        path.status().message +
        ". A C0 path is not an answer here -- at a kink q_a'' is undefined, the path-velocity "
        "limit collapses to sigma_dot = 0 and the machine stops dead at every waypoint"));
  }
  plan.path = std::move(path).value();

  // 4.5, step three, and the reason it is a step and not a formality: the
  // shortcut replaced detours by their chords and the fit is not those chords,
  // so what the search cleared is not what would be flown.
  auto rechecked = check_path(
    model, plan.path, *request.collision_scene, request.payload, request.payload_shape,
    collision);
  if (!rechecked.ok()) {
    return Result<SamplingPlan>::failure(rechecked.status());
  }
  plan.recheck = std::move(rechecked).value();
  if (!plan.recheck.clear) {
    return Result<SamplingPlan>::failure(
      refuse(
        ErrorCode::InvalidArgument,
        "the smoothed path is blocked where the sampled one was clear, which is the case 4.5's "
        "re-check exists for: the shortcut replaced a detour by its chord and the C2 fit runs "
        "inside the box its waypoints span rather than along them, so the curve that would be "
        "flown is not the polyline that was cleared. " + describe(plan.recheck)));
  }

  return Result<SamplingPlan>::success(std::move(plan));
}

}  // namespace crane_planning
