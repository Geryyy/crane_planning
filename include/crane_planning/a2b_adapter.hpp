// `a2b_movement` (`timber_crane_planning_interfaces/CalcMovement`), as a thin
// adapter over `/crane/plan_motion`.
//
// # Why the service is kept and the message is not
//
// `wiki/implementation/ros2_interfaces.md` 9: `a2b_movement` is the interface the
// timber workflow depends on and both stacks call today, so the new planner also
// serves it. Its payload fields are `wood_log_msgs/LogShape` and `Log[]` -- a
// cylinder description -- which is why a concrete block is currently declared to
// the legacy planner **as a cylinder**. Keeping the service compatible is worth an
// adapter; carrying `LogShape` into the new stack is not. So the fiction stops
// here: everything below this boundary speaks `crane_msgs/Payload`, and a block on
// the native path stays a box.
//
// # It is an adapter and not a planner
//
// Nothing here plans, limits, or checks a collision. `translate_a2b_request`
// produces a `crane_msgs/PlanMotion::Request`, `PlannerNode::plan` answers it --
// the same call `/crane/plan_motion` is answered by, so the same start state, the
// same scene, the same kappa, the same latency budget and the same
// `/crane/reference` publication -- and `translate_a2b_response` carries the
// answer back. There is no second set of limits and no second collision
// configuration, because there is no second planner.
//
// # What `CalcMovement` says and what it does not
//
// Two things the `.srv` carries only as a comment, and that the callers settle:
//
//   * **The frame.** `CalcMovement` has no `header` and therefore no frame field.
//     All three callers convert into `K0_mounting_base` before they call --
//     `mp_rviz_panel` transforms into its `toFrameRel_`, which is that string;
//     `check_pose_feasibility.py` applies its own `world_to_k0_*` and sends the
//     result; `epsilon_crane_behavior_tree` passes through a `yN` that was built
//     in it -- and the legacy server answers in it, stamping its published path
//     `K0_mounting_base`. The adapter therefore **asserts** that frame and
//     converts nothing, so ROS 2 Interfaces 4's "frame conversions happen in
//     exactly two places" stays true with the two places unchanged.
//
//   * **The body.** `y_n` is documented as the "target position of tip (K5)" --
//     the pivot the pendulum hangs from, `K5_inner_telescope` -- while
//     `crane_msgs/PlanMotion::goal` is the **tool** pose, `K8_tool_center_point`.
//     `mp_rviz_panel` says the same thing in code, adding `d_0508 + d_contact` to
//     the height a user types before it sends it. The two points are one drop
//     apart and `tip_to_tcp_drop` reads that drop out of the description rather
//     than writing it down.

#ifndef CRANE_PLANNING__A2B_ADAPTER_HPP_
#define CRANE_PLANNING__A2B_ADAPTER_HPP_

#include <string>

#include "crane_model/model.hpp"
#include "crane_msgs/msg/payload.hpp"
#include "crane_msgs/srv/plan_motion.hpp"
#include "timber_crane_planning_interfaces/srv/calc_movement.hpp"

namespace crane_planning
{

/// The name the retained callers resolve, kept exactly as they resolve it.
/**
 * ROS 2 Interfaces 1 puts *new* cross-node contracts under `/crane/...`; this is
 * not a new one. `mp_rviz_panel` and every timber behaviour tree ask for the
 * relative `a2b_movement` from a node in the root namespace and
 * `check_pose_feasibility.py` asks for `/a2b_movement` outright, so the absolute
 * name all three land on is this one. Renaming it would be retiring the service
 * rather than keeping it compatible.
 */
inline constexpr char kA2bMovementService[] = "/a2b_movement";

/// How far the TCP hangs below the tip pivot K5 with the tool at rest, m.
/**
 * `y_n` names the pivot and `PlanMotion::goal` names the tool, so the adapter
 * needs the distance between them. It is a property of the description and the
 * mounted tool and is read out of the model at the hanging equilibrium of an
 * empty gripper -- never written down, because the PZS100's rail gripper and the
 * 7040's jaw do not hang the same distance.
 *
 * It is a **vertical drop** and not a vector: `wiki/trajectory_planning.md` 6
 * makes the endpoint a genuine steady state of the passive subsystem, and a tool
 * whose centre of mass is on its own axis comes to rest hanging straight down
 * from the pivot whatever the arm is doing. A payload whose centre of mass is
 * off that axis tilts the pendulum, and then the drop is the vertical part of a
 * hang that also has a horizontal one -- see the note on `translate_a2b_request`.
 */
[[nodiscard]] bool tip_to_tcp_drop(
  const crane_model::Model & model, double & drop_m, std::string & why);

/// The payload half of the mapping: `LogShape` and its two centres, or a refusal.
/**
 * Separate from `translate_a2b_request` -- which calls it, so the whole mapping
 * is still readable in one place -- because the caller of the adapter needs the
 * payload before it needs anything else.
 *
 * `carries_log == false` is `SHAPE_NONE` and nothing else is read: the timber
 * trees send a `logCarrying` shape on an *approach* move as well, and it
 * describes the log they are going to pick up rather than one they are holding.
 */
[[nodiscard]] bool translate_a2b_payload(
  const timber_crane_planning_interfaces::srv::CalcMovement::Request & request,
  crane_msgs::msg::Payload & payload, std::string & why);

/// Every field of `CalcMovement::Request`, mapped or refused by name.
/**
 * | `CalcMovement` | `crane_msgs/PlanMotion` | rule |
 * |---|---|---|
 * | `y_n` | `goal.pose.position` | the tip pivot, lowered by `tip_to_tcp_drop_m` onto the tool |
 * | (no field) | `goal.header.frame_id` | `K0_mounting_base`, asserted and not converted |
 * | `phi_tool_n` | `goal.pose.orientation` | `phi_z` about K0's z, scalar-last on the wire |
 * | `slow_down` | `speed_scale` | `1 / slow_down`; a divider below one is refused |
 * | `carries_log`, `log_carrying`, `m_log`, | `payload` | `translate_a2b_payload` |
 * | `s_log_8`, `coll_shape`, `p_cyl_8` | | below |
 * | `check_log_collision`, | `avoid_collisions` | one flag for one flag |
 * | `check_gripper_collision` | | below |
 * | `logs_scene` | -- | refused: obstacles arrive on `/crane/collision_scene` |
 * | `q0`, `q0_dot` | -- | refused unless the `.srv` default: the start is measured |
 * | `t_end` | -- | refused unless zero: the OCP derives the duration |
 * | `v_d_tip` | -- | refused unless zero: the move ends at rest |
 * | `publish_path` | -- | carried by the response's `tcp_path`, not by a topic |
 *
 * The one thing that is bounded rather than exact is a **tapered** log. A
 * `wood_log_msgs/LogShape` is a truncated cone and `crane_msgs/Payload` carries
 * one radius, so the shape becomes the cone's **enclosing** cylinder at
 * `max(radius_top, radius_bottom)` -- deliberately not the legacy mean, which
 * under-covers the wider end. The mass is the caller's own `m_log` and is not
 * recomputed from it.
 *
 * The other is the hang: `tip_to_tcp_drop_m` is the drop of an empty gripper, so
 * a request that carries a payload whose centre of mass is off the tool axis
 * lands the pivot within the pendulum's horizontal hang of `y_n` rather than on
 * it. That is the same convention the legacy interface has always had -- its
 * `d_contact` is a constant too -- and it is stated rather than hidden.
 */
[[nodiscard]] bool translate_a2b_request(
  const timber_crane_planning_interfaces::srv::CalcMovement::Request & request,
  double tip_to_tcp_drop_m, crane_msgs::srv::PlanMotion::Request & plan, std::string & why);

/// The answer, back. Three fields, and the third of them is why this is a table.
/**
 * `success` is **`bool` on both sides**, and that is worth saying because ROS 2
 * Interfaces 1's `[!warning]` records that two service types in the retained
 * stack disagree about it. The two are `CalcMovement::Response::success`, which
 * is `bool`, and `CalcGripMovement::Response::success`, which is `int64`. Only
 * the first is `a2b_movement`, which is the one service 9 names, so what this
 * adapter reconciles is `bool` to `bool` with the same meaning on both sides --
 * and the `int64` belongs to a service it does not serve. The legacy server
 * nevertheless assigns integers into that `bool` field (`response->success = 0`,
 * `success == 1`), so the disagreement is real in the code even where the two
 * types agree; here the field is written from `PlanMotion`'s `bool` and from
 * nothing else.
 *
 * A refusal carries **no trajectory**, unlike `/crane/plan_motion`, which hands
 * back whatever is still standing on `/crane/reference` and says so in
 * `message`. `CalcMovement::Response` has no `message` field, so a caller could
 * not tell "the previous plan is still running" from "here is something new";
 * the legacy server left the trajectory empty on failure and so does this.
 */
void translate_a2b_response(
  const crane_msgs::srv::PlanMotion::Response & plan,
  timber_crane_planning_interfaces::srv::CalcMovement::Response & response);

}  // namespace crane_planning

#endif  // CRANE_PLANNING__A2B_ADAPTER_HPP_
