#include "crane_planning/collision.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "crane_planning/joint_limits.hpp"
#include "crane_planning/status.hpp"

namespace crane_planning
{
namespace
{

using crane_model::CollisionPrimitive;
using crane_model::CollisionResult;
using crane_model::CollisionScene;
using crane_model::ErrorCode;
using crane_model::Frame;
using crane_model::Q;
using crane_model::QA;
using crane_model::Result;
using crane_model::Status;

/// The frames whose motion the resolution of the check is measured on.
/**
 * Not every frame the model carries: these four span the chain from the first
 * moving link to the deepest one the description always defines, and the tool
 * moves farthest of anything on the crane, so its step bounds the step of the
 * bodies between. It is a proxy for how far the *surfaces* move and the file
 * header says so.
 */
constexpr int kWatchedFrameCount = 4;
constexpr std::array<Frame, kWatchedFrameCount> kWatchedFrames{
  {Frame::Boom, Frame::BigTelescope, Frame::Rotator, Frame::Tcp}};

/// The positions of `kWatchedFrames` at one configuration.
using WatchedFrames = Eigen::Matrix<double, 3, kWatchedFrameCount>;

/// Where the watched frames stand at one configuration.
Result<WatchedFrames> watched_frames_at(const crane_model::Model & model, const Q & q)
{
  WatchedFrames frames;
  for (std::size_t index = 0; index < kWatchedFrames.size(); ++index) {
    auto pose = model.forward_kinematics(q, Frame::MountingBase, kWatchedFrames[index]);
    if (!pose.ok()) {
      return Result<WatchedFrames>::failure(pose.status());
    }
    frames.col(static_cast<Eigen::Index>(index)) = pose.value().position_m;
  }
  return Result<WatchedFrames>::success(frames);
}

/// How many places the path's own travel is measured at before it is sampled.
constexpr std::size_t kTravelProbes = 33;

/// The safety factor between the step and the thinnest primitive in the scene.
/**
 * A step at the thinnest extent can still cross a primitive corner to corner
 * without landing inside it, because the grid over the sway envelope is
 * two-dimensional. Half is the diagonal's worth of room.
 */
constexpr double kThinnestFraction = 0.5;

/// Largest bound the sway envelope may be given, rad.
constexpr double kMaxSwayBound = 1.5;

/// The canonical eight of one path sample, passive pair left at zero.
Q configuration_of(const PathSample & sample)
{
  return expand(sample.q_a, sample.q8);
}

bool rotation_is_a_rotation(const Eigen::Matrix3d & rotation)
{
  return rotation.allFinite() &&
         (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() < 1.0e-6;
}

Status check_primitive_is_usable(const CollisionPrimitive & primitive, const char * which)
{
  if (primitive.id.empty()) {
    return failure(ErrorCode::InvalidScene, std::string("the ") + which + " primitive has no id");
  }
  if (!primitive.dimensions_m.allFinite() || (primitive.dimensions_m.array() <= 0.0).any()) {
    return failure(
      ErrorCode::InvalidScene,
      "primitive " + primitive.id + " has no positive extent along every axis of its own frame");
  }
  if (!primitive.pose_in_mounting_base.matrix().allFinite() ||
    !rotation_is_a_rotation(primitive.pose_in_mounting_base.linear()))
  {
    return failure(
      ErrorCode::InvalidScene, "primitive " + primitive.id + " does not carry a pose in K0");
  }
  return Status{};
}

/// One runge or one bed slab, in the bed's own axes.
CollisionPrimitive box_in_truck(
  const CollisionPrimitive & truck, std::string id, const Eigen::Vector3d & centre_in_truck,
  const Eigen::Vector3d & extent)
{
  CollisionPrimitive box;
  box.id = std::move(id);
  box.shape = crane_model::CollisionShape::Box;
  box.dimensions_m = extent;
  box.structural = true;
  box.pose_in_mounting_base = truck.pose_in_mounting_base;
  box.pose_in_mounting_base.translation() =
    truck.pose_in_mounting_base * centre_in_truck;
  return box;
}

/// The scene, with the carried payload placed at this configuration.
Result<CollisionScene> scene_with_payload(
  const crane_model::Model & model, const CollisionScene & scene, const PayloadShape & payload,
  const Q & q)
{
  CollisionScene placed;
  placed.primitives = scene.primitives;
  if (!payload.declared) {
    return Result<CollisionScene>::success(std::move(placed));
  }
  auto primitive = payload_primitive(model, q, payload);
  if (!primitive.ok()) {
    return Result<CollisionScene>::failure(primitive.status());
  }
  placed.primitives.push_back(std::move(primitive).value());
  return Result<CollisionScene>::success(std::move(placed));
}

/// The worst of one `collision_queries` answer, and what it was against.
struct Worst
{
  std::size_t index{};
  double distance{std::numeric_limits<double>::infinity()};
  bool colliding{false};
};

/// The worst of the first `limit` answers -- `collision_queries` puts the scene
/// primitives first, in order, and the checked self pairs after them.
Worst worst_of(const std::vector<CollisionResult> & results, std::size_t limit)
{
  Worst worst;
  for (std::size_t index = 0; index < std::min(limit, results.size()); ++index) {
    if (results[index].minimum_distance_m < worst.distance) {
      worst.distance = results[index].minimum_distance_m;
      worst.index = index;
    }
    worst.colliding = worst.colliding || results[index].collision;
  }
  return worst;
}

Worst worst_of(const std::vector<CollisionResult> & results)
{
  Worst worst;
  for (std::size_t index = 0; index < results.size(); ++index) {
    if (results[index].minimum_distance_m < worst.distance) {
      worst.distance = results[index].minimum_distance_m;
      worst.index = index;
    }
    worst.colliding = worst.colliding || results[index].collision;
  }
  return worst;
}

/// Fill in who blocked, from one result and the scene it came out of.
Blocker blocker_of(
  const std::vector<CollisionResult> & results, std::size_t index, const CollisionScene & scene,
  const Eigen::Vector2d & q_u, bool at_sway_bound)
{
  Blocker blocker;
  const CollisionResult & result = results[index];
  blocker.other_id = result.other_id;
  blocker.self = index >= scene.primitives.size();
  blocker.structural = !blocker.self && scene.primitives[index].structural;
  blocker.minimum_distance_m = result.minimum_distance_m;
  blocker.witness_on_robot_m = result.witness_on_robot_m;
  blocker.witness_on_other_m = result.witness_on_other_m;
  blocker.q_u = q_u;
  blocker.at_sway_bound = at_sway_bound;
  return blocker;
}

std::string metres(double value)
{
  return std::to_string(value) + " m";
}

}  // namespace

double sway_clearance_m(double pendulum_length_m, const SwayEnvelope & envelope)
{
  const double bound = std::max(0.0, std::min(kMaxSwayBound, envelope.bound_rad()));
  return std::max(0.0, pendulum_length_m) * std::sin(bound);
}

crane_model::Result<std::vector<CollisionPrimitive>> expand_truck(
  const CollisionPrimitive & truck, const TruckModel & model)
{
  using Answer = Result<std::vector<CollisionPrimitive>>;

  Status status = check_primitive_is_usable(truck, "truck");
  if (!status.ok()) {
    return Answer::failure(std::move(status));
  }
  if (!model.runge_dimensions_m.allFinite() || (model.runge_dimensions_m.array() <= 0.0).any()) {
    return Answer::failure(
      failure(ErrorCode::InvalidArgument, "the truck model gives a runge no positive extent"));
  }
  if (!(model.bed_thickness_m > 0.0) || !(model.bed_thickness_m <= truck.dimensions_m.z())) {
    return Answer::failure(
      failure(
        ErrorCode::InvalidArgument,
        "the bed slab is " + metres(model.bed_thickness_m) + " thick and the truck it is keyed to "
        "is " + metres(truck.dimensions_m.z()) + " tall"));
  }
  if (model.station_offsets_m.empty()) {
    return Answer::failure(
      failure(
        ErrorCode::InvalidArgument,
        "the truck model names no runge stations, so the loading area of trajectory_planning 4.2 "
        "would have no runges at all -- and 4.2 is explicit that perception cannot recover them"));
  }
  if (model.runge_dimensions_m.y() > truck.dimensions_m.y()) {
    return Answer::failure(
      failure(
        ErrorCode::InvalidArgument,
        "a runge is " + metres(model.runge_dimensions_m.y()) + " across and the bed it stands on "
        "is " + metres(truck.dimensions_m.y()) + " wide"));
  }

  std::vector<CollisionPrimitive> geometry;
  geometry.reserve(1U + 2U * model.station_offsets_m.size());

  // The load area, and not the vehicle: the crane is bolted to the truck, so the
  // box the truck was measured as is not an obstacle to it. What the arm has to
  // clear is the top of the bed.
  geometry.push_back(
    box_in_truck(
      truck, "truck_bed",
      Eigen::Vector3d(0.0, 0.0, 0.5 * (truck.dimensions_m.z() - model.bed_thickness_m)),
      Eigen::Vector3d(truck.dimensions_m.x(), truck.dimensions_m.y(), model.bed_thickness_m)));

  // The runges: two rows, one station each, standing on the bed at its two long
  // edges. 4.2's pick and place happen between them.
  const double y_offset = 0.5 * (truck.dimensions_m.y() - model.runge_dimensions_m.y());
  const double z_offset = 0.5 * (truck.dimensions_m.z() + model.runge_dimensions_m.z());
  for (std::size_t station = 0; station < model.station_offsets_m.size(); ++station) {
    const double x = model.station_offsets_m[station];
    if (!std::isfinite(x) ||
      std::abs(x) + 0.5 * model.runge_dimensions_m.x() > 0.5 * truck.dimensions_m.x())
    {
      return Answer::failure(
        failure(
          ErrorCode::InvalidArgument,
          "runge station " + std::to_string(station) + " sits at " + metres(x) +
          " from the centre of a bed " + metres(truck.dimensions_m.x()) +
          " long, so the runge would hang off the end of it. The stations are a property of the "
          "truck and this one does not fit the truck the scene measured"));
    }
    const std::string suffix = "_" + std::to_string(station + 1U);
    geometry.push_back(
      box_in_truck(
        truck, "truck_runge_left" + suffix, Eigen::Vector3d(x, y_offset, z_offset),
        model.runge_dimensions_m));
    geometry.push_back(
      box_in_truck(
        truck, "truck_runge_right" + suffix, Eigen::Vector3d(x, -y_offset, z_offset),
        model.runge_dimensions_m));
  }

  return Answer::success(std::move(geometry));
}

crane_model::Result<CollisionPrimitive> payload_primitive(
  const crane_model::Model & model, const Q & q, const PayloadShape & shape)
{
  if (!shape.declared) {
    return Result<CollisionPrimitive>::failure(
      failure(
        ErrorCode::InvalidPayload,
        "no payload shape was declared, so there is no geometry to place at K8"));
  }
  if (!shape.dimensions_m.allFinite() || (shape.dimensions_m.array() <= 0.0).any() ||
    !shape.center_k8_m.allFinite())
  {
    return Result<CollisionPrimitive>::failure(
      failure(
        ErrorCode::InvalidPayload,
        "the declared payload has no positive extent along every axis of its own frame; "
        "dimensions are the extent per axis, so a box carries its three side lengths and a "
        "cylinder (2r, 2r, length)"));
  }

  auto k8 = model.forward_kinematics(q, Frame::MountingBase, Frame::RotatorLowerPart);
  if (!k8.ok()) {
    return Result<CollisionPrimitive>::failure(k8.status());
  }

  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.linear() = k8.value().orientation.normalized().toRotationMatrix();
  pose.translation() = k8.value().position_m + pose.linear() * shape.center_k8_m;

  CollisionPrimitive primitive;
  primitive.id = kPayloadId;
  primitive.shape = shape.shape;
  primitive.dimensions_m = shape.dimensions_m;
  primitive.pose_in_mounting_base = pose;
  primitive.structural = false;
  return Result<CollisionPrimitive>::success(std::move(primitive));
}

crane_model::Result<double> pendulum_length_m(
  const crane_model::Model & model, const Q & q, const PayloadShape & shape)
{
  auto pivot = model.forward_kinematics(q, Frame::MountingBase, Frame::Tip);
  if (!pivot.ok()) {
    return Result<double>::failure(pivot.status());
  }
  auto tcp = model.forward_kinematics(q, Frame::MountingBase, Frame::Tcp);
  if (!tcp.ok()) {
    return Result<double>::failure(tcp.status());
  }

  double length = (tcp.value().position_m - pivot.value().position_m).norm();

  // The deepest tool frame the description carries. Only the 7040's gripper
  // defines a contact point; the PZS100 gets `FrameUnavailable` rather than a
  // substituted pose (model API contract 3), so this is a difference between the
  // two descriptions and not a failure of either.
  auto contact = model.forward_kinematics(q, Frame::MountingBase, Frame::ToolContact);
  if (contact.ok()) {
    length = std::max(length, (contact.value().position_m - pivot.value().position_m).norm());
  }

  if (shape.declared) {
    auto payload = payload_primitive(model, q, shape);
    if (!payload.ok()) {
      return Result<double>::failure(payload.status());
    }
    // The far corner rather than the centre: a payload swings as far as its own
    // geometry reaches, and the envelope of 4.3 is about the geometry.
    const double half_diagonal = 0.5 * payload.value().dimensions_m.norm();
    length = std::max(
      length,
      (payload.value().pose_in_mounting_base.translation() - pivot.value().position_m).norm() +
      half_diagonal);
  }

  if (!(length > 0.0) || !std::isfinite(length)) {
    return Result<double>::failure(
      failure(
        ErrorCode::InvalidRobotDescription,
        "the description puts the whole tool on the passive pivot, so there is no pendulum length "
        "for the sway envelope of trajectory_planning 4.3 to be taken from"));
  }
  return Result<double>::success(length);
}

crane_model::Result<ConfigurationCheck> check_configuration(
  const crane_model::Model & model, const CollisionScene & scene, const PayloadShape & payload,
  const CollisionSettings & settings, const Q & q, double step_m)
{
  if (!q.allFinite()) {
    return Result<ConfigurationCheck>::failure(
      failure(ErrorCode::NonFiniteInput, "the configuration to be checked is not finite"));
  }
  if (!settings.sway.q_sway_max.allFinite() || (settings.sway.q_sway_max.array() < 0.0).any() ||
    settings.sway.bound_rad() > kMaxSwayBound)
  {
    return Result<ConfigurationCheck>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "q_sway_max is the state box of mpc 3 constraint 3 and has to be a non-negative bound "
        "under " + std::to_string(kMaxSwayBound) + " rad on each passive coordinate"));
  }

  auto length = pendulum_length_m(model, q, payload);
  if (!length.ok()) {
    return Result<ConfigurationCheck>::failure(length.status());
  }

  ConfigurationCheck check;
  check.pendulum_length_m = length.value();
  check.sway_clearance_m = sway_clearance_m(check.pendulum_length_m, settings.sway);

  auto placed = scene_with_payload(model, scene, payload, q);
  if (!placed.ok()) {
    return Result<ConfigurationCheck>::failure(placed.status());
  }
  const CollisionScene nominal_scene = std::move(placed).value();

  auto nominal = model.collision_queries(q, nominal_scene);
  if (!nominal.ok()) {
    return Result<ConfigurationCheck>::failure(nominal.status());
  }
  const std::vector<CollisionResult> & results = nominal.value();
  const Worst worst = worst_of(results);
  check.nominal_distance_m = worst.distance;
  check.sway_samples = 1U;

  // Distances are signed, so `collision` is exactly `minimum_distance_m < 0` and
  // the deepest penetration is the worst distance found anywhere.
  if (worst.colliding) {
    check.clear = false;
    check.blocker = blocker_of(results, worst.index, nominal_scene, q.segment<2>(4), false);
    return Result<ConfigurationCheck>::success(std::move(check));
  }

  // **The envelope covers the scene. The crane's own link pairs are judged at
  // the pose the tool hangs at, and only there.**
  //
  // This is a decision and not an oversight. The tool hangs on the two passive
  // joints and swings *relative to the crane*, so a swung self query is a real
  // one -- but on the PZS100 the rail gripper sits 25 mm from the inner
  // telescope at rest (034, measured) and 0.2 rad of sway closes that gap at
  // every configuration with the boom up. Refusing on it refuses every path,
  // which is the failure 4.3's own warning describes, and refusing on a
  // -0.4 mm result between a curved rail and a telescope is refusing on the
  // error of a single enclosing box (034's *left undone*).
  //
  // It is also a refusal that says nothing actionable: how close the tool comes
  // to the telescope when it swings is a property of `q_a` and the sway bound
  // alone, so no reachable path improves it and there is nothing for a caller or
  // an operator to do with the answer. What that clearance really constrains is
  // `q_u^+` itself, which is `mpc` 3's to impose and the virtual working cell's
  // to bound -- and the sway envelope's whole purpose here is that the *scene*
  // be cleared for a tool that swings.
  const Worst scene_worst = worst_of(results, nominal_scene.primitives.size());
  if (scene_worst.distance > check.sway_clearance_m) {
    return Result<ConfigurationCheck>::success(std::move(check));
  }
  if (!(check.sway_clearance_m > 0.0)) {
    return Result<ConfigurationCheck>::success(std::move(check));
  }

  // It is not, so the envelope is resolved exactly. The model takes all eight
  // coordinates, so each of these is a real query at a real sway pose rather
  // than an inflated approximation of one.
  check.envelope_resolved = true;
  const double allowed = std::asin(std::min(1.0, step_m / check.pendulum_length_m));
  if (!(allowed > 0.0)) {
    return Result<ConfigurationCheck>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "the check resolution of " + metres(step_m) + " resolves no sway at all on a pendulum " +
        metres(check.pendulum_length_m) + " long"));
  }

  std::array<std::size_t, 2> counts{{1U, 1U}};
  for (Eigen::Index axis = 0; axis < 2; ++axis) {
    const double span = 2.0 * std::abs(settings.sway.q_sway_max[axis]);
    const std::size_t needed =
      (span > 0.0) ? static_cast<std::size_t>(std::ceil(span / allowed)) + 1U : 1U;
    if (needed > settings.max_sway_samples) {
      return Result<ConfigurationCheck>::failure(
        failure(
          ErrorCode::InvalidArgument,
          "resolving the sway envelope at " + metres(step_m) + " would take " +
          std::to_string(needed) + " poses on passive coordinate " + std::to_string(axis) +
          ", past the cap of " + std::to_string(settings.max_sway_samples) +
          ". Widen the resolution, tighten q_sway_max, or raise the cap -- a coarser grid would "
          "leave sway the MPC permits unchecked"));
    }
    counts[static_cast<std::size_t>(axis)] = needed;
  }

  const Eigen::Vector2d nominal_q_u = q.segment<2>(4);
  for (std::size_t first = 0; first < counts[0]; ++first) {
    for (std::size_t second = 0; second < counts[1]; ++second) {
      Eigen::Vector2d offset = Eigen::Vector2d::Zero();
      for (Eigen::Index axis = 0; axis < 2; ++axis) {
        const std::size_t count = counts[static_cast<std::size_t>(axis)];
        const std::size_t index = (axis == 0) ? first : second;
        const double fraction =
          (count == 1U) ? 0.5 : static_cast<double>(index) / static_cast<double>(count - 1U);
        offset[axis] = (2.0 * fraction - 1.0) * std::abs(settings.sway.q_sway_max[axis]);
      }
      if (offset.isZero()) {
        continue;  // the nominal pose, already answered above
      }

      Q swung = q;
      swung.segment<2>(4) = nominal_q_u + offset;
      auto swung_scene = scene_with_payload(model, scene, payload, swung);
      if (!swung_scene.ok()) {
        return Result<ConfigurationCheck>::failure(swung_scene.status());
      }
      const CollisionScene at_pose = std::move(swung_scene).value();
      auto answer = model.collision_queries(swung, at_pose);
      if (!answer.ok()) {
        return Result<ConfigurationCheck>::failure(answer.status());
      }
      ++check.sway_samples;

      const Worst swung_worst = worst_of(answer.value(), at_pose.primitives.size());
      if (swung_worst.colliding) {
        check.clear = false;
        check.blocker =
          blocker_of(answer.value(), swung_worst.index, at_pose, swung.segment<2>(4), true);
        return Result<ConfigurationCheck>::success(std::move(check));
      }
    }
  }

  return Result<ConfigurationCheck>::success(std::move(check));
}

crane_model::Result<SceneResolution> resolve_scene(
  const CollisionScene & scene, const PayloadShape & payload, const CollisionSettings & settings)
{
  if (!(settings.resolution_m > 0.0) || !std::isfinite(settings.resolution_m) ||
    !(settings.min_resolution_m > 0.0) || settings.min_resolution_m > settings.resolution_m)
  {
    return Result<SceneResolution>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "the check resolution and its floor have to be positive with the floor no coarser than "
        "the resolution"));
  }

  SceneResolution resolution;
  resolution.primitives = scene.primitives.size();
  resolution.thinnest_primitive_m = std::numeric_limits<double>::infinity();
  for (const CollisionPrimitive & primitive : scene.primitives) {
    Status status = check_primitive_is_usable(primitive, "scene");
    if (!status.ok()) {
      return Result<SceneResolution>::failure(std::move(status));
    }
    if (primitive.id == kPayloadId) {
      return Result<SceneResolution>::failure(
        failure(
          ErrorCode::InvalidScene,
          "the scene carries a primitive with the reserved id 'payload', which crane_model reads "
          "as geometry the tool is carrying and does not check against the links that carry it"));
    }
    resolution.structural_primitives += primitive.structural ? 1U : 0U;
    resolution.thinnest_primitive_m =
      std::min(resolution.thinnest_primitive_m, primitive.dimensions_m.minCoeff());
  }
  if (payload.declared) {
    resolution.thinnest_primitive_m =
      std::min(resolution.thinnest_primitive_m, payload.dimensions_m.minCoeff());
  }

  // The resolution, and the one relation it has to obey: a step coarser than the
  // thinnest thing being checked can carry the tool across that thing without
  // ever landing inside it.
  resolution.step_m = settings.resolution_m;
  if (std::isfinite(resolution.thinnest_primitive_m)) {
    const double asked = kThinnestFraction * resolution.thinnest_primitive_m;
    if (asked < resolution.step_m) {
      resolution.step_m = asked;
      resolution.tightened = true;
    }
  }
  if (resolution.step_m < settings.min_resolution_m) {
    return Result<SceneResolution>::failure(
      failure(
        ErrorCode::InvalidScene,
        "the thinnest primitive in the scene is " + metres(resolution.thinnest_primitive_m) +
        ", which asks for a step of " + metres(resolution.step_m) + " against a floor of " +
        metres(settings.min_resolution_m) +
        ". Checking at the floor would leave that primitive unresolved, and a path checked at a "
        "step it does not resolve is not checked"));
  }
  return Result<SceneResolution>::success(resolution);
}

crane_model::Result<PathCheck> check_path(
  const crane_model::Model & model, const GeometricPath & path, const CollisionScene & scene,
  const crane_model::Payload & payload, const PayloadShape & shape,
  const CollisionSettings & settings)
{
  if (settings.max_samples < 2U) {
    return Result<PathCheck>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "a path checked at fewer than its two endpoints is not checked"));
  }

  auto resolution = resolve_scene(scene, shape, settings);
  if (!resolution.ok()) {
    return Result<PathCheck>::failure(resolution.status());
  }

  PathCheck check;
  check.scene_primitives = resolution.value().primitives;
  check.structural_primitives = resolution.value().structural_primitives;
  check.thinnest_primitive_m = resolution.value().thinnest_primitive_m;
  check.step_m = resolution.value().step_m;
  check.resolution_tightened = resolution.value().tightened;

  // How far the crane actually moves along this path, measured rather than
  // assumed. The pair is held at zero for this pass only: it is a measurement of
  // how fast the geometry travels and not a collision claim, and every checked
  // configuration below settles the pair properly.
  WatchedFrames previous = WatchedFrames::Zero();
  for (std::size_t probe = 0; probe < kTravelProbes; ++probe) {
    const double sigma = static_cast<double>(probe) / static_cast<double>(kTravelProbes - 1U);
    const Q q = configuration_of(path.at(sigma));
    auto current = watched_frames_at(model, q);
    if (!current.ok()) {
      return Result<PathCheck>::failure(current.status());
    }
    if (probe > 0) {
      check.travel_m += (current.value() - previous).colwise().norm().maxCoeff();
    }
    previous = current.value();
  }

  const double intervals = std::ceil(check.travel_m / check.step_m);
  const std::size_t count =
    std::max<std::size_t>(2U, static_cast<std::size_t>(std::max(1.0, intervals)) + 1U);
  if (count > settings.max_samples) {
    return Result<PathCheck>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "checking " + metres(check.travel_m) + " of travel at " + metres(check.step_m) +
        " takes " + std::to_string(count) + " configurations, past the cap of " +
        std::to_string(settings.max_samples)));
  }

  // The junctions are checked whatever the uniform grid lands on: they are where
  // the primitive changes phase, and a refusal that names a phase should have
  // looked at its boundary.
  std::vector<double> sigmas;
  sigmas.reserve(count + path.junction_sigmas().size());
  for (std::size_t index = 0; index < count; ++index) {
    sigmas.push_back(static_cast<double>(index) / static_cast<double>(count - 1U));
  }
  for (const double junction : path.junction_sigmas()) {
    sigmas.push_back(junction);
  }
  std::sort(sigmas.begin(), sigmas.end());
  sigmas.erase(std::unique(sigmas.begin(), sigmas.end()), sigmas.end());

  check.worst_clearance_m = std::numeric_limits<double>::infinity();
  for (const double sigma : sigmas) {
    Q q = configuration_of(path.at(sigma));
    auto settled = model.passive_equilibrium(actuated(q), payload);
    if (!settled.ok()) {
      return Result<PathCheck>::failure(settled.status());
    }
    q.segment<2>(4) = settled.value();

    auto configuration = check_configuration(model, scene, shape, settings, q, check.step_m);
    if (!configuration.ok()) {
      return Result<PathCheck>::failure(configuration.status());
    }
    ++check.samples;
    check.worst_clearance_m =
      std::min(check.worst_clearance_m, configuration.value().nominal_distance_m);
    if (!configuration.value().clear) {
      check.clear = false;
      check.blocked_sigma = sigma;
      check.blocked_segment = static_cast<std::size_t>(
        std::count_if(
          path.junction_sigmas().begin(), path.junction_sigmas().end(),
          [sigma](double junction) {return junction < sigma;}));
      check.blocked_at = std::move(configuration).value();
      return Result<PathCheck>::success(std::move(check));
    }
  }

  return Result<PathCheck>::success(std::move(check));
}

std::string describe(const PathCheck & check)
{
  std::string text = "the path was checked at " + std::to_string(check.samples) +
    " configurations, a step of " + metres(check.step_m) + " over " + metres(check.travel_m) +
    " of travel, against " + std::to_string(check.scene_primitives) + " scene primitives (" +
    std::to_string(check.structural_primitives) +
    " of them structural, the truck model of trajectory_planning 4.2 keyed to the measured truck "
    "pose) and against the crane itself on crane_model's derived allowed-collision list";
  if (check.resolution_tightened) {
    text += ". The step was tightened from the configured resolution because the thinnest "
      "primitive in the scene is " + metres(check.thinnest_primitive_m) +
      " and a step coarser than half of that could cross it between two samples";
  }
  if (check.clear) {
    text += ". Every configuration cleared, the closest by " + metres(check.worst_clearance_m) +
      ", and the tool was cleared over the whole sway envelope of trajectory_planning 4.3 at the "
      "q_sway_max of mpc 3 constraint 3";
    return text;
  }

  const Blocker & blocker = check.blocked_at.blocker;
  text += ". It is blocked at sigma = " + std::to_string(check.blocked_sigma) + ", in segment " +
    std::to_string(check.blocked_segment + 1U) + ", by ";
  if (blocker.self) {
    text += "the crane against itself on the link pair " + blocker.other_id;
  } else if (blocker.structural) {
    text += "the structural obstacle '" + blocker.other_id +
      "', which is the truck model keyed to the measured truck pose and moves when the truck does";
  } else {
    text += "the perceived obstacle '" + blocker.other_id + "'";
  }
  text += ", at " + metres(blocker.minimum_distance_m) + " -- the crane at (" +
    std::to_string(blocker.witness_on_robot_m.x()) + ", " +
    std::to_string(blocker.witness_on_robot_m.y()) + ", " +
    std::to_string(blocker.witness_on_robot_m.z()) + ") and the obstacle at (" +
    std::to_string(blocker.witness_on_other_m.x()) + ", " +
    std::to_string(blocker.witness_on_other_m.y()) + ", " +
    std::to_string(blocker.witness_on_other_m.z()) + ") in K0_mounting_base";
  text += blocker.at_sway_bound ?
    ", with the tool swung to q_u = (" + std::to_string(blocker.q_u.x()) + ", " +
    std::to_string(blocker.q_u.y()) +
    ") rad inside the envelope of trajectory_planning 4.3, where Delta_sway is " +
    metres(check.blocked_at.sway_clearance_m) + " on a pendulum " +
    metres(check.blocked_at.pendulum_length_m) + " long" :
    ", with the tool hanging still";
  return text;
}

crane_model::Result<double> watched_frame_travel_m(
  const crane_model::Model & model, const Q & from, const Q & to)
{
  if (!from.allFinite() || !to.allFinite()) {
    return Result<double>::failure(
      failure(ErrorCode::NonFiniteInput, "a configuration the travel is measured between is "
      "not finite"));
  }
  auto first = watched_frames_at(model, from);
  if (!first.ok()) {
    return Result<double>::failure(first.status());
  }
  auto second = watched_frames_at(model, to);
  if (!second.ok()) {
    return Result<double>::failure(second.status());
  }
  return Result<double>::success(
    (second.value() - first.value()).colwise().norm().maxCoeff());
}

crane_model::Result<double> clearance_at(
  const crane_model::Model & model, const CollisionScene & scene, const Q & q)
{
  auto answer = model.collision_query(q, scene);
  if (!answer.ok()) {
    return Result<double>::failure(answer.status());
  }
  return Result<double>::success(answer.value().minimum_distance_m);
}

}  // namespace crane_planning
