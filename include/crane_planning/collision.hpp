// Collision checking for the geometric stage of `wiki/trajectory_planning.md` 4:
// the scene the planner is told about, the truck the scene is keyed to, the
// crane against itself, and the sway envelope of 4.3.
//
// # One model, one library, one URDF -- and none of it is here
//
// 4.2's simplification is that collision comes from the same Pinocchio geometry
// the kinematics does, with Coal answering the queries. That is `crane_model`'s
// backend and this file does not re-implement any of it: every distance below is
// a `Model::collision_query` or `Model::collision_queries` call, so the geometry
// the planner refuses a path against is the geometry the model was built from
// and there is no second collision model to drift.
//
// What is left for the planner is the part 4.2 and 4.3 give it, and it is
// exactly three things: the scene, the truck, and the sway.
//
// # The sway envelope, and what actually ships
//
// 4.3 asks that the path be cleared for a tool that swings, with the envelope
//
//     Delta_sway = l_tool sin(max|q_u^+|)
//
// and the **same** bound `q_u^+` that `wiki/mpc.md` 3 constraint 3 imposes as a
// state box, so that any sway the MPC permits is sway the path was cleared for
// by construction rather than by coincidence. That bound lives in exactly one
// place in this workspace and it is this package's `q_sway_max` parameter; the
// virtual working cell of `wiki/control_architecture.md` 5.1 consumes the same
// number and, by that section's own rule, derives its envelope independently of
// the planner's data.
//
// 4.3's *mechanism* was to inflate the tool and payload geometry by
// `Delta_sway`. Issue 034 could not implement that -- the frozen model API has
// no inflation parameter and adding one is an architecture change -- and its
// notes left two alternatives on the record: require
// `minimum_distance_m > Delta_sway`, or query at the sway bound itself, since
// the model takes all eight coordinates and a query *at* the bound is a real
// query. Neither alternative works alone on this machine:
//
//   * `minimum_distance_m` is the smallest distance found **anywhere** -- over
//     all fifteen links and, for the self result, over every checked link pair.
//     Demanding `Delta_sway` of it demands half a metre of clearance between the
//     mounting base and the truck bed it is bolted to, and between the rail
//     gripper and the inner telescope, which sit 25 mm apart at the neutral
//     configuration (034, measured). That is not conservatism, it is a planner
//     that refuses every path. It also refuses the one move 4.2 exists for:
//     picking **between** the runges, which are 2.12 m tall and about a metre
//     apart.
//   * Querying the envelope alone is exact where it is sampled and silent
//     between samples, and a 2-D grid over the sway box costs N^2 queries at
//     every checked configuration along the path.
//
// So both ship, and each does the half it is good at. At every checked
// configuration the nominal (hanging) pose is queried first, against the scene
// and the crane's own link pairs alike. If the smallest distance to the *scene*
// then exceeds `Delta_sway`, **no admissible sway can bring the tool into
// contact with it** -- that is 4.3's inflate-by-`Delta_sway` test, used as a
// sufficient condition, and it settles the free-space majority in one query.
// Only where it fails is the envelope gridded and queried exactly, at the
// resolution below. The wiki page carries this and the reason for it; see 4.3.
//
// # The envelope covers the scene; the crane covers itself at the hanging pose
//
// Self-collision is checked at every configuration, and at the pose the tool
// really hangs at. It is deliberately **not** re-checked at each sway pose, for
// three reasons that all point the same way:
//
//   * The machine has no room for it. The PZS100's rail gripper sits 25 mm from
//     the inner telescope at rest, and 0.2 rad of sway closes that at every
//     configuration with the boom up. A planner that refuses on it refuses
//     everything, which is the failure 4.3's own warning describes.
//   * The geometry does not support it. One enclosing box per link is 034's
//     *left undone*; a -0.4 mm result between a curved rail and a telescope is
//     inside the fit's own error, not a collision anyone measured.
//   * There is nothing to do with the answer. How close the tool comes to the
//     crane when it swings is a property of `q_a` and the sway bound alone, so
//     no reachable path improves it. What it really constrains is `q_u^+`, which
//     is `mpc` 3's to impose and the virtual working cell's to bound.
//
// The sway envelope's purpose in 4.3 is that the *scene* be cleared for a tool
// that swings, and that is exactly what it does here.
//
// # A stated resolution, and its relation to the smallest checked primitive
//
// A path checked only at its waypoints is not checked, and neither is a sway
// envelope checked only at its corners. Both are sampled at `resolution_m`,
// measured as the largest distance any of the model frames this file watches
// moves between two neighbouring checked configurations.
//
// The relation to the geometry is the reason the number is what it is: a step
// larger than the thinnest checked primitive can carry the tool from one side of
// that primitive to the other with no sample inside it, and a 2-D grid can do it
// diagonally, so the step is held at or below **half** the thinnest extent in
// the scene. A configured resolution coarser than that is tightened to it and
// the answer says so; the tightening is bounded by `min_resolution_m`, and a
// scene that would need a finer step than that is refused rather than checked at
// a step that does not resolve it. This is a sampling rule and not a proof: the
// frozen API answers for a configuration, not for the sweep between two, so
// continuous collision detection is not available to be used here.
//
// # What is not here
//
// The sampling fallback a blocked primitive falls back to is
// `sampling_planner.hpp`; what is here is the check both mechanisms are cleared
// by, and `watched_frame_travel_m` below is the resolution rule shared with it.
// Coal's broadphase is `crane_model`'s and is issue 034's *left undone*:
// `collision_queries` is O(links x primitives + pairs), and the shortcut above is
// what keeps this affordable without one.

#ifndef CRANE_PLANNING__COLLISION_HPP_
#define CRANE_PLANNING__COLLISION_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstddef>
#include <string>
#include <vector>

#include "crane_model/model.hpp"
#include "crane_planning/geometric_path.hpp"

namespace crane_planning
{

/// The scene id `crane_model` reserves for a payload the tool is carrying.
/**
 * The frozen collision signature takes `q` and a scene and no `Payload`, so a
 * carried payload reaches the model as a scene primitive at the K0 pose the
 * caller reads out of `forward_kinematics` (`crane_model`'s README, "A carried
 * payload"). Placing it is this package's half; not checking it against the
 * bodies the rotator joint carries is the model's.
 */
inline constexpr char kPayloadId[] = "payload";

/// The scene id this planner reserves for the measured pose of the truck.
/**
 * `trajectory_planning` 4.2: the truck bed and the runges are **structural**
 * obstacles, a property of the vehicle rather than of perception, and they move
 * when it does. So the world model publishes where the truck is -- one primitive
 * with this id, its pose and its extent -- and the planner expands it into the
 * bed and the six runges through `expand_truck`. That keeps the geometry keyed
 * to the measured pose without anybody hard-coding a position in K0.
 */
inline constexpr char kTruckId[] = "truck";

/// The admissible sway bound of `wiki/mpc.md` 3 constraint 3, `[tip, tilt]`, rad.
/**
 * `q_u^+` of `wiki/nomenclature.md` 8, spelled `q_sway_max` there. Constraint 3
 * is `|q_u - q_eq| <= q_u^+`, a box around the *equilibrium* and not around
 * zero, which is why everything below offsets the grid from
 * `Model::passive_equilibrium` rather than from the origin.
 */
struct SwayEnvelope
{
  Eigen::Vector2d q_sway_max{Eigen::Vector2d::Zero()};

  /// `max|q_u^+|`, the bound `Delta_sway` is taken at.
  [[nodiscard]] double bound_rad() const noexcept {return q_sway_max.cwiseAbs().maxCoeff();}
};

/// `Delta_sway = l_tool sin(max|q_u^+|)`, m -- 4.3's envelope, as a length.
[[nodiscard]] double sway_clearance_m(double pendulum_length_m, const SwayEnvelope & envelope);

/// The payload's *shape*, which `crane_model::Payload` does not carry.
/**
 * `crane_model::Payload` is a mass, a centre of mass and an inertia: everything
 * the equilibrium needs and nothing a collision query can use. `crane_msgs/
 * Payload` carries the shape and the extent beside them, so this is the other
 * half of the same message, converted at the node boundary.
 *
 * `dimensions_m` is the extent along each axis of the primitive's own frame, the
 * convention `crane_model` fixes and `ros2_interfaces` 6 now states: a box
 * carries its three side lengths, a cylinder `(2r, 2r, length)`.
 */
struct PayloadShape
{
  bool declared{false};
  crane_model::CollisionShape shape{crane_model::CollisionShape::Box};
  Eigen::Vector3d dimensions_m{Eigen::Vector3d::Zero()};
  Eigen::Vector3d center_k8_m{Eigen::Vector3d::Zero()};  ///< where it sits in K8
};

/// The truck, as geometry rather than as three numbers in a planner YAML.
/**
 * 4.2 is explicit that pick and place happen **between the runges**, that they
 * are thin, tall and directly in the descent corridor, and that perception
 * cannot be relied on to recover them. The legacy planner carried them
 * hard-coded as `postsRight`/`postsLeft`, two rows of three boxes of
 * 0.28 x 0.31 x 2.12 m, in a motion-planning YAML together with the bed.
 *
 * The **dimensions are preserved and the hard-coding is dropped**: what is
 * configured here is a property of the vehicle -- how big a runge is and where
 * along the bed the three stations are -- and where it all ends up in K0 comes
 * from the measured pose of the `truck` primitive. Move the truck and the
 * runges move with it.
 */
struct TruckModel
{
  /// A runge's extent in the bed's own axes: along, across, up. Legacy, m.
  Eigen::Vector3d runge_dimensions_m{0.28, 0.31, 2.12};

  /// Where the three stations sit along the bed, from its centre, m.
  std::vector<double> station_offsets_m{-2.0, 0.0, 2.0};

  /// How thick a slab the load area is represented by, m.
  /**
   * The `truck` primitive's extent is the vehicle's, and the crane is bolted to
   * the front of it: checking the mounting base against the whole box would
   * report the crane colliding with the truck it is part of. What the arm has to
   * clear is the **load area**, so the bed enters the scene as a slab of this
   * thickness at the top face of that box, and the vehicle below it is not an
   * obstacle to a crane mounted on it.
   */
  double bed_thickness_m{0.10};
};

/// Expand the measured truck pose into the structural geometry of 4.2.
/**
 * `truck` is the primitive the scene carries with `kTruckId`: its pose is the
 * centre of the vehicle box and its `dimensions_m` are that box's extent along
 * its own axes, with x along the bed and z up.
 *
 * Out come seven primitives, every one of them `structural = true`: the bed
 * slab, and six runges standing on it at the two long edges, three to a side.
 * A station that would put a runge off the end of the bed is refused naming both
 * numbers, because a truck model that does not fit the truck it was keyed to is
 * a model of a different vehicle.
 */
[[nodiscard]] crane_model::Result<std::vector<crane_model::CollisionPrimitive>> expand_truck(
  const crane_model::CollisionPrimitive & truck, const TruckModel & model);

/// The payload as the scene primitive the reserved id stands for.
/**
 * At the K8 pose of the configuration it is asked for, offset by the payload's
 * own centre of mass in K8, which is where `crane_msgs/Payload` puts it.
 */
[[nodiscard]] crane_model::Result<crane_model::CollisionPrimitive> payload_primitive(
  const crane_model::Model & model, const crane_model::Q & q, const PayloadShape & shape);

/// The pendulum length `l_tool` of 4.3, m, at one configuration.
/**
 * Measured from the passive pivot -- the tilt frame, which is the second of the
 * two passive joints -- to the farthest carried point: the deepest tool frame
 * the description defines, and the far corner of the payload when one is
 * carried. It is a length the description and the payload decide, not a
 * configured one, so the two tools and every load give their own.
 */
[[nodiscard]] crane_model::Result<double> pendulum_length_m(
  const crane_model::Model & model, const crane_model::Q & q, const PayloadShape & shape);

/// Everything a deployment configures about the check.
struct CollisionSettings
{
  SwayEnvelope sway{};

  /// Largest step between two checked configurations, m. See the file header.
  double resolution_m{0.10};

  /// Floor on the step the scene may tighten it to, m.
  double min_resolution_m{0.01};

  /// Cap on the configurations one path may be checked at.
  std::size_t max_samples{2048};

  /// Cap on the grid the sway envelope is resolved with, per passive axis.
  std::size_t max_sway_samples{15};

  TruckModel truck{};
};

/// What a scene decides about the step a check will run at, before any of it runs.
struct SceneResolution
{
  double step_m{};                   ///< the resolution actually used
  bool tightened{false};             ///< the scene asked for a finer step than configured
  double thinnest_primitive_m{};     ///< smallest extent in the checked scene
  std::size_t primitives{};
  std::size_t structural_primitives{};
};

/// The step, and the scene validation that decides it. **The one place.**
/**
 * Split out of `check_path` because the sampling fallback has to subdivide its
 * motions at the same step, and a second derivation of "the resolution" is a
 * second answer waiting to drift. The rule is the file header's: the configured
 * `resolution_m`, tightened to at most half the thinnest extent anything in the
 * scene has, and a scene that would need a step finer than `min_resolution_m` is
 * refused rather than checked at a step that does not resolve it.
 *
 * A primitive with no id, no pose or no positive extent, and the reserved
 * `payload` id in a scene the planner did not place it in, are refusals here --
 * which is where they were before, and is before any query has been run.
 */
[[nodiscard]] crane_model::Result<SceneResolution> resolve_scene(
  const crane_model::CollisionScene & scene, const PayloadShape & payload,
  const CollisionSettings & settings);

/// Which obstacle blocked, and where the two bodies were closest.
struct Blocker
{
  std::string other_id;  ///< the scene primitive, or `first|second` for a self pair
  bool self{false};      ///< the crane against itself, on `crane_model`'s allowed list
  bool structural{false};    ///< the truck model rather than the perception stream
  double minimum_distance_m{};
  Eigen::Vector3d witness_on_robot_m{Eigen::Vector3d::Zero()};
  Eigen::Vector3d witness_on_other_m{Eigen::Vector3d::Zero()};
  Eigen::Vector2d q_u{Eigen::Vector2d::Zero()};  ///< the sway pose it was found at
  bool at_sway_bound{false};  ///< false when the nominal, hanging pose was enough
};

/// One configuration, checked at its nominal pose and over its sway envelope.
struct ConfigurationCheck
{
  bool clear{true};
  double pendulum_length_m{};
  double sway_clearance_m{};      ///< `Delta_sway` at this configuration
  double nominal_distance_m{};    ///< the smallest distance found at the hanging pose
  std::size_t sway_samples{};     ///< envelope poses queried; 1 is nominal only
  bool envelope_resolved{false};  ///< whether the grid had to be walked at all
  Blocker blocker{};              ///< meaningful when `clear` is false
};

/// Check one configuration. `q` carries its own passive pair, at equilibrium.
[[nodiscard]] crane_model::Result<ConfigurationCheck> check_configuration(
  const crane_model::Model & model, const crane_model::CollisionScene & scene,
  const PayloadShape & payload, const CollisionSettings & settings, const crane_model::Q & q,
  double step_m);

/// How far the frames this file watches move between two configurations, m.
/**
 * The quantity `resolution_m` is measured in, exposed because the sampling
 * fallback of `sampling_planner.hpp` has to subdivide its motions at the **same**
 * stated resolution -- a fallback validated at its endpoints and a primitive
 * sampled every `resolution_m` would not be cleared on the same terms, which is
 * the whole point of reusing this check.
 *
 * It is a straight-line measure between two configurations and not a path
 * integral: the caller is expected to subdivide until the answer is under the
 * step, which is what makes it the same rule `check_path` applies along a fitted
 * path. The passive pair is taken as each argument carries it, so a caller
 * measuring how fast the geometry travels may leave it at zero exactly as
 * `check_path`'s own travel probe does.
 */
[[nodiscard]] crane_model::Result<double> watched_frame_travel_m(
  const crane_model::Model & model, const crane_model::Q & from, const crane_model::Q & to);

/// One path, checked along its length.
struct PathCheck
{
  bool clear{true};
  std::size_t samples{};        ///< configurations checked, endpoints included
  double step_m{};              ///< the resolution actually used
  double travel_m{};            ///< how far the watched frames move along the path
  double thinnest_primitive_m{};  ///< smallest extent in the checked scene
  bool resolution_tightened{false};  ///< the scene asked for a finer step than configured
  std::size_t scene_primitives{};
  std::size_t structural_primitives{};
  double worst_clearance_m{};   ///< smallest nominal distance seen anywhere on the path
  double blocked_sigma{};       ///< where it was blocked; meaningful when `clear` is false
  std::size_t blocked_segment{};  ///< which segment that sigma falls in
  ConfigurationCheck blocked_at{};
};

/// Check a whole path, at the resolution of the file header, and say what blocked.
/**
 * The passive pair is solved per sample with `Model::passive_equilibrium`, which
 * is what makes the nominal pose the pose the tool really hangs at and is the
 * expensive part of a check. `payload` is the carried mass the equilibrium needs
 * and `shape` is the geometry the scene needs; the two are separate because
 * `crane_model::Payload` carries no shape.
 */
[[nodiscard]] crane_model::Result<PathCheck> check_path(
  const crane_model::Model & model, const GeometricPath & path,
  const crane_model::CollisionScene & scene, const crane_model::Payload & payload,
  const PayloadShape & shape, const CollisionSettings & settings);

/// The check, as the sentence a refusal or a response carries.
[[nodiscard]] std::string describe(const PathCheck & check);

/// The scene's own clearance at one configuration, for the redundancy score.
/**
 * `wiki/robot_model.md` 2.2 step 3 scores the telescope redundancy by joint-range
 * centring **and** collision clearance. This is the second half: the smallest
 * distance the whole crane has to anything, self included, at one configuration.
 * It is a single `collision_query` and it does not resolve the sway envelope --
 * a preference between two extensions is not the safety check, and the safety
 * check is `check_path`.
 */
[[nodiscard]] crane_model::Result<double> clearance_at(
  const crane_model::Model & model, const crane_model::CollisionScene & scene,
  const crane_model::Q & q);

}  // namespace crane_planning

#endif  // CRANE_PLANNING__COLLISION_HPP_
