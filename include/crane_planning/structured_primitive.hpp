// The structured primitive of `wiki/trajectory_planning.md` 4.4: **lift,
// traverse, descend**, tried first because most crane moves are exactly that.
//
// 4.4 gives the order and the reason. The primitive is cheap, deterministic and
// smooth by construction, and it is tried before the sampling planner because a
// sampling planner is stochastic and its runtime is not bounded -- a poor fit
// for the re-plan loop of 7. Primitive-first buys deterministic latency in the
// common case and leaves completeness to the rare one. It also means the
// ordinary trajectory is smooth and predictable, which matters for an operator
// watching a large machine move.
//
// 4.4's order is **generate, check, accept**, and all three happen inside
// `build_structured_primitive` as one step. The check is `collision.hpp`'s:
// the path is sampled at a stated resolution, every sample is put through
// `Model::collision_query` against `/crane/collision_scene` plus the structural
// truck geometry of 4.2, the tool is cleared over the sway envelope of 4.3, and
// a blocked sample refuses the primitive naming the phase it fell in.
//
// # The transfer altitude comes from the scene
//
// The legacy planner hard-codes it, and 9's open items ask for it to be derived
// instead. `derive_transfer_altitude` is that derivation and is the only place
// the altitude is decided. It reads the two endpoints and -- when there is one
// -- the obstacle extent, and a configured value may bound it from below or from
// above but may not *be* it.
//
// The obstacle term's input is `/crane/collision_scene`, through
// `scene_extent`. The absence of a scene is still a **named gap** and not a
// zero: `SceneExtent` carries `obstacles_known == false` and a NaN extent when
// nothing has been received or the scene is empty, so the term cannot be read as
// "there are no obstacles", and `scene_without_obstacles()` is the one place the
// planner says so.
//
// # What is out of scope here, and where it is
//
// A blocked primitive is a **refusal** here and never a fallback: 4.4's second
// mechanism lives in `sampling_planner.hpp` and the order between the two is
// `planner_core.hpp`'s to enforce, so nothing in this file knows that a fallback
// exists. The time parametrization, the sway-carrying OCP, the force and flow
// constraints and kappa are issue 043, so this file produces geometry and no
// timing; `/crane/plan_grip`'s own descend/close/open/lift phases are issue 044
// and are a different set on a different time base -- the descend phase here is
// the arm's, not the gripper's.

#ifndef CRANE_PLANNING__STRUCTURED_PRIMITIVE_HPP_
#define CRANE_PLANNING__STRUCTURED_PRIMITIVE_HPP_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include "crane_model/model.hpp"
#include "crane_planning/arm_geometry.hpp"
#include "crane_planning/collision.hpp"
#include "crane_planning/geometric_path.hpp"
#include "crane_planning/inverse_kinematics.hpp"
#include "crane_planning/joint_limits.hpp"

namespace crane_planning
{

/// The three phases of 4.4's primitive, in the order it names them.
enum class PrimitivePhase : std::uint8_t
{
  Lift = 0,
  Traverse = 1,
  Descend = 2
};

inline constexpr std::size_t kPrimitivePhaseCount = 3;

/// The phase's name, as every refusal spells it.
[[nodiscard]] const char * phase_name(PrimitivePhase phase) noexcept;

/// What the planner knows about the scene the transfer altitude is derived from.
/**
 * One field is the obstacle extent and one field is whether there is an obstacle
 * extent at all, because those are different facts and only the second is known
 * today. `highest_obstacle_z_m` is NaN while `obstacles_known` is false so that
 * an unguarded read propagates rather than quietly reading as ground level.
 */
struct SceneExtent
{
  bool obstacles_known{false};
  double highest_obstacle_z_m{std::numeric_limits<double>::quiet_NaN()};
};

/// The scene when nothing has been received on `/crane/collision_scene`.
/**
 * The gap, stated rather than defaulted: no extent is known, so the obstacle
 * term of the derivation does not run and cannot be read as "there are no
 * obstacles". A plan that asked for collision avoidance is refused before it
 * gets here; a plan that did not gets this and an altitude that clears its two
 * endpoints and nothing else.
 */
[[nodiscard]] SceneExtent scene_without_obstacles();

/// The obstacle extent of a scene that *has* been received.
/**
 * The tallest point of any primitive, measured as the top of its own box
 * projected onto K0's z axis, so a rotated primitive is measured at the height
 * it really reaches. An **empty** scene is still `obstacles_known == false`:
 * a world model that has seen nothing and a world model that has published
 * nothing are the same statement about the transfer altitude, which is that
 * there is no obstacle for it to clear.
 */
[[nodiscard]] SceneExtent scene_extent(const crane_model::CollisionScene & scene);

/// A floor and a ceiling a deployment may put on the derivation. Neither is it.
/**
 * Both are absent by default, so the shipped behaviour is the derivation alone.
 * A floor may only raise the derived altitude and a ceiling may only lower it,
 * which is what keeps a configured number from becoming the answer: with no
 * endpoints and no scene there is no altitude at all, whatever is configured.
 */
struct TransferAltitudeSettings
{
  std::optional<double> floor_m{};
  std::optional<double> ceiling_m{};
};

/// How deep into the tool the description let the clearance be measured.
/**
 * The two machines are not described to the same depth and the model API
 * contract 3 says so outright: only the 7040's gripper defines a
 * `tool_contact_point`, and `Frame::ToolContact` is `FrameUnavailable` on the
 * PZS100 rather than a substituted pose. So the clearance is measured to the
 * deepest tool frame the description actually carries, and which one that was
 * travels on the answer instead of being inferred from the tool.
 */
enum class ToolReference : std::uint8_t
{
  ToolContact,  ///< the gripper's own contact point -- the jaw
  ToolCentre    ///< the TCP, which is as deep as the description goes -- the rail gripper
};

[[nodiscard]] const char * tool_reference_name(ToolReference reference) noexcept;

/// The derived transfer altitude, with every term that produced it.
struct TransferAltitude
{
  double z_m{};                     ///< the altitude the traverse runs at, K0, m
  double z_from_endpoints_m{};      ///< what the two endpoints alone asked for
  double z_tcp_start_m{};
  double z_tcp_goal_m{};
  double tool_reach_start_m{};      ///< rotator bearing to the deepest tool frame, start, m
  double tool_reach_goal_m{};       ///< and at the goal -- a different number per tool
  ToolReference reference_start{ToolReference::ToolCentre};
  ToolReference reference_goal{ToolReference::ToolCentre};
  bool obstacle_term_applied{false};  ///< false when no scene carried an extent
  double z_from_obstacles_m{std::numeric_limits<double>::quiet_NaN()};
  bool floor_binding{false};
  bool ceiling_binding{false};
};

/// Derive the transfer altitude from the endpoints and the scene. **The one place.**
/**
 * The endpoint term raises the TCP by the length of the tool assembly hanging
 * below the rotator bearing, measured from the model at that endpoint's own
 * configuration:
 *
 *     z = max over endpoints of ( z_tcp + | p_tool - p_rotator | )
 *
 * so at the transfer altitude everything below the rotator sits at or above the
 * altitude the TCP held at the higher endpoint. Whatever the tool was reaching
 * down into when it was there -- the stack it placed on, the truck bed, the
 * block it picked -- is cleared by the tool's whole described length before the
 * traverse begins, and the number is a property of the mounted tool read off the
 * description rather than of a configuration file. It is a different number for
 * the rail gripper and for the jaw, which is the point: the descend phase's
 * clearance geometry is the tool's.
 *
 * `p_tool` is the deepest tool frame the description carries -- see
 * `ToolReference`. It is **not** the rail's or the jaw's tip: those live in links
 * the frozen `Frame` enum does not name, so this clearance is a lower bound on
 * the tool's true vertical envelope. The envelope itself is the collision
 * geometry `crane_model` fitted, and what the altitude does not have to cover
 * the check of `collision.hpp` does: an altitude that clears the endpoints and
 * leaves the traverse in something is refused, not flown.
 *
 * The obstacle term is the same statement about the tallest thing in the way,
 * and it is applied only when `scene.obstacles_known` -- see `scene_extent` and
 * `scene_without_obstacles()`.
 *
 * The tool reaches are the caller's forward-kinematics results rather than a
 * model handle, so that this function is the derivation and nothing else, and a
 * test can put any pair of endpoints through it without a machine.
 */
[[nodiscard]] crane_model::Result<TransferAltitude> derive_transfer_altitude(
  double z_tcp_start_m, double tool_reach_start_m, double z_tcp_goal_m, double tool_reach_goal_m,
  const SceneExtent & scene, const TransferAltitudeSettings & settings);

/// The reach of the tool assembly below the rotator bearing, at one configuration.
/**
 * Split out of the derivation because it is the one part that needs a model, and
 * because it is where the two tools stop being described to the same depth:
 * `Frame::ToolContact` exists on the 7040's jaw and is `FrameUnavailable` on the
 * PZS100's rail gripper, so `reference` says which frame the number was measured
 * to rather than leaving a caller to assume.
 */
struct ToolReach
{
  double reach_m{};
  ToolReference reference{ToolReference::ToolCentre};
  double z_tcp_m{};
  double phi_z{};  ///< the endpoint's own yaw, which the waypoint above it keeps
};

[[nodiscard]] crane_model::Result<ToolReach> measure_tool_reach(
  const crane_model::Model & model, const crane_model::Q & q);

/// The derivation in one sentence, for the service response.
[[nodiscard]] std::string describe(const TransferAltitude & altitude);

/// The middle step of 4.4's generate, check, accept.
struct PrimitiveCheck
{
  bool clear{true};
  bool checked{false};  ///< false when the caller asked for a collision-blind plan
  PrimitivePhase phase{PrimitivePhase::Lift};  ///< meaningful only when `clear` is false
  std::string note;                            ///< what was and was not checked
  PathCheck path{};                            ///< the check itself, when one ran
};

/// One move the primitive is asked to cover.
struct PrimitiveRequest
{
  crane_model::Q q_start{crane_model::Q::Zero()};  ///< all eight, passive pair as measured
  crane_model::Q q_goal{crane_model::Q::Zero()};   ///< the endpoint issue 039 solved

  /// The measured rate at the start, over the five path coordinates of 4.1.
  /**
   * Carried straight to `PathFitRequest::start_rate`, which is where the reason
   * for it is. Zero is the stopped start of `wiki/trajectory_planning.md` 7 and
   * is what a machine standing still gives; anything else makes the lift phase
   * leave its first waypoint at the velocity the machine already has.
   */
  PathVector dq_start{PathVector::Zero()};

  crane_model::Payload payload{};
  PayloadShape payload_shape{};  ///< the other half of `crane_msgs/Payload`

  /// The extent the transfer altitude is derived against.
  /**
   * Ignored when `collision_scene` is set: the extent is then derived from that
   * scene, so the altitude and the check cannot be told about two different
   * worlds.
   */
  SceneExtent scene{};

  /// `avoid_collisions` of `crane_msgs/PlanMotion`, carried down to the check.
  bool avoid_collisions{true};

  /// The scene the check runs against, already in `K0_mounting_base`.
  /**
   * Null means none has been received. With `avoid_collisions` that is a
   * refusal, because a plan that reads as checked and was not is worse than no
   * plan; without it the primitive is built and the answer says nothing was
   * checked.
   */
  const crane_model::CollisionScene * collision_scene{nullptr};
};

/// What a deployment configures about the primitive.
struct PrimitiveSettings
{
  TransferAltitudeSettings altitude{};
  PathFitSettings fit{};
  CollisionSettings collision{};
};

/// Check the generated primitive against the scene, the truck and itself.
/**
 * 4.4's middle step, and `collision.hpp` does the work: the path is sampled at a
 * stated resolution, every sample is checked at the pose the tool hangs in and
 * over the sway envelope of 4.3, and the first blocked sample refuses the
 * primitive naming the phase its sigma fell in. A failure here is a check that
 * could not be run -- an unusable scene, a description with no collision
 * geometry -- and is not the same answer as a path that was checked and blocked.
 */
[[nodiscard]] crane_model::Result<PrimitiveCheck> check_primitive(
  const crane_model::Model & model, const GeometricPath & path,
  const PrimitiveRequest & request, const PrimitiveSettings & settings);

/// The primitive, generated, checked and accepted.
struct StructuredPrimitive
{
  GeometricPath path{};
  TransferAltitude altitude{};
  PrimitiveCheck check{};
  IkSolution lift_waypoint{};      ///< the configuration the lift phase ends at
  IkSolution traverse_waypoint{};  ///< the configuration the traverse phase ends at
};

/// Generate, check and accept the primitive -- or refuse it, naming the phase.
/**
 * Refused and never deformed. A start or a goal outside the joint range, a
 * transfer altitude that ends up below both endpoints, a waypoint the arm cannot
 * reach, a segment that will not interpolate monotonically: each of these
 * returns a failure whose message names which of lift, traverse and descend
 * could not be built. There is no fallback *here*: 4.4's second mechanism is
 * `sampling_planner.hpp` and whether a refusal becomes one is `plan_motion`'s
 * decision, which is what keeps 4.4's order in one place.
 *
 * The two interior waypoints are solved with the **semi-analytic** route of
 * `wiki/robot_model.md` 2.2 rather than with the equilibrium-constrained NLP.
 * 2.2's closing paragraph is the rule: the constrained route is for an endpoint
 * that must be sway-free, and these two are transit configurations the machine
 * passes through rather than stops at. The endpoint that does have to be
 * sway-free is `request.q_goal`, which the caller has already solved that way.
 */
[[nodiscard]] crane_model::Result<StructuredPrimitive> build_structured_primitive(
  const crane_model::Model & model, const ArmGeometry & geometry, const JointLimits & limits,
  const IkSettings & ik, const PrimitiveSettings & settings, const PrimitiveRequest & request);

}  // namespace crane_planning

#endif  // CRANE_PLANNING__STRUCTURED_PRIMITIVE_HPP_
