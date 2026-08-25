#include "crane_planning/planner_core.hpp"

#include <cstddef>
#include <string>
#include <utility>

namespace crane_planning
{
namespace
{

using crane_model::ErrorCode;
using crane_model::Frame;
using crane_model::Q;
using crane_model::Result;
using crane_model::Status;

Status failure(ErrorCode code, std::string message)
{
  return Status{code, std::move(message)};
}

}  // namespace

crane_model::Result<PlannerContext> build_planner(
  const crane_model::Model & model, const std::string & robot_description_xml,
  const PlannerSettings & settings)
{
  auto limits = read_joint_limits(robot_description_xml, model.urdf_joint_names());
  if (!limits.ok()) {
    return Result<PlannerContext>::failure(limits.status());
  }
  auto geometry = probe_arm_geometry(
    model, limits.value(), settings.geometry_samples, settings.ik.geometry);
  if (!geometry.ok()) {
    return Result<PlannerContext>::failure(geometry.status());
  }

  PlannerContext context;
  context.limits = std::move(limits).value();
  context.geometry = std::move(geometry).value();
  context.settings = settings;
  return Result<PlannerContext>::success(std::move(context));
}

crane_model::Result<MotionPlan> plan_motion(
  const crane_model::Model & model, const PlannerContext & context,
  const MotionRequest & request)
{
  // `crane_msgs/PlanMotion` defaults `avoid_collisions` to true, so a caller
  // that fills nothing in asks for a checked plan -- and gets a refusal rather
  // than a trajectory that reads as checked when no scene has arrived.
  if (request.avoid_collisions && request.scene == nullptr) {
    return Result<MotionPlan>::failure(
      failure(
        ErrorCode::NotReady,
        "avoid_collisions is set and nothing has been received on /crane/collision_scene, so "
        "there is no scene to check against. The truck bed and the runges of "
        "trajectory_planning 4.2 are keyed to the truck pose that scene carries, so they are not "
        "known either. Set avoid_collisions=false to ask for a plan that says in as many words "
        "that nothing was checked"));
  }
  {
    Status status = check_margin_factor(request.margin_factor);
    if (!status.ok()) {
      return Result<MotionPlan>::failure(std::move(status));
    }
  }
  if (!request.q_a_start.allFinite()) {
    return Result<MotionPlan>::failure(
      failure(ErrorCode::NonFiniteInput, "the start configuration is not finite"));
  }

  IkRequest goal;
  goal.p_tcp_0 = request.p_tcp_0;
  goal.phi_z_d = request.phi_z_d;
  // The tool coordinate is held where the machine has it. q8 is not a path
  // variable (trajectory_planning 4.1); what opens and closes the gripper is
  // `/crane/plan_grip`, which is issue 044.
  goal.q8 = request.q_a_start[static_cast<Eigen::Index>(kToolRow)];
  goal.payload = request.payload;
  // The clearance half of robot_model 2.2 step 3's redundancy score, which is a
  // preference between telescope extensions rather than a check. It is offered
  // the scene whenever there is one: `avoid_collisions` governs whether the path
  // is *checked*, and a well-cleared extension is not a worse answer to a caller
  // who asked for no check.
  goal.scene = request.scene;

  // The route decision of robot_model 2.2's closing paragraph, made here rather
  // than asked of the caller: a `/crane/plan_motion` goal is a *placement* goal,
  // so its endpoint has to be a genuine steady state of the passive subsystem
  // and it is solved with the equilibrium-constrained NLP. The semi-analytic
  // route is 2.2's "where speed matters" one and it still runs -- inside this
  // call, as the initial guess.
  auto endpoint = solve_equilibrium_constrained_ik(
    model, context.geometry, context.limits, context.settings.ik,
    context.settings.equilibrium, goal);
  if (!endpoint.ok()) {
    return Result<MotionPlan>::failure(endpoint.status());
  }

  MotionPlan plan;
  plan.endpoint = std::move(endpoint).value();

  // The start as all eight coordinates. The passive pair is where it hangs at
  // the measured actuated configuration -- the stopped-start convention this
  // package still carries and that trajectory_planning 7 calls a defect, lifted
  // by issue 045.
  auto settled_start = model.passive_equilibrium(request.q_a_start, request.payload);
  if (!settled_start.ok()) {
    return Result<MotionPlan>::failure(settled_start.status());
  }
  PrimitiveRequest primitive;
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    primitive.q_start[static_cast<Eigen::Index>(kActuatedRows[row])] =
      request.q_a_start[static_cast<Eigen::Index>(row)];
  }
  primitive.q_start[4] = settled_start.value()[0];
  primitive.q_start[5] = settled_start.value()[1];
  primitive.q_goal = plan.endpoint.q;
  primitive.payload = request.payload;
  primitive.payload_shape = request.payload_shape;
  primitive.avoid_collisions = request.avoid_collisions;
  // The scene, once, for both the transfer altitude and the check. Without one
  // the altitude clears the two endpoints and says so, and the check either was
  // not asked for or has already been refused above.
  primitive.collision_scene = request.scene;
  primitive.scene = scene_without_obstacles();

  // trajectory_planning 4.4's order, and the whole of it: generate the
  // primitive, check it, accept it if clear -- and only then, and only because it
  // was not, sample. Nothing below runs the fallback beside the primitive or to
  // compare with it; 4.4's [!important] says the deterministic common case is
  // what primitive-first buys, and a fallback that runs anyway spends it.
  auto built = build_structured_primitive(
    model, context.geometry, context.limits, context.settings.ik, context.settings.primitive,
    primitive);
  if (built.ok()) {
    plan.mechanism = PathMechanism::StructuredPrimitive;
    plan.primitive = std::move(built).value();
    plan.path = plan.primitive.path;
  } else {
    plan.primitive_refusal = built.status().message;

    // There is a fallback only when there is something to sample around. A
    // collision-blind request never gets here -- an unchecked primitive is never
    // blocked -- and a checked request without a scene was refused at the top of
    // this function, so what is left is a primitive that could not be *built*
    // for a request nobody asked to be checked. Sampling that would be sampling
    // against nothing.
    if (!request.avoid_collisions || request.scene == nullptr) {
      return Result<MotionPlan>::failure(built.status());
    }

    SamplingRequest sampling;
    sampling.q_start = primitive.q_start;
    sampling.q_goal = primitive.q_goal;
    sampling.payload = request.payload;
    sampling.payload_shape = request.payload_shape;
    sampling.collision_scene = request.scene;

    auto sampled = plan_sampled_path(
      model, context.limits, context.settings.primitive.fit,
      context.settings.primitive.collision, context.settings.sampling, sampling);
    if (!sampled.ok()) {
      // Both refusals, in the order they happened. A caller told only that the
      // search timed out cannot tell whether the primitive was blocked by a
      // runge or was never reachable at all.
      return Result<MotionPlan>::failure(
        failure(
          sampled.status().code,
          plan.primitive_refusal + ". So the fallback ran, and " + sampled.status().message));
    }
    plan.mechanism = PathMechanism::SamplingFallback;
    plan.sampled = std::move(sampled).value();
    plan.path = plan.sampled.path;
  }

  auto trajectory = scaled_ramp_along_path(
    plan.path, context.limits, request.margin_factor, context.settings.ramp);
  if (!trajectory.ok()) {
    return Result<MotionPlan>::failure(trajectory.status());
  }
  plan.trajectory = std::move(trajectory).value();

  // The visualisation path, and it is only that: the row it fills in the
  // response is documented `for visualization only`. It is where the tool hangs
  // at each sampled configuration, so it shows the lift and the descend as the
  // operator will see them and it is not what any check ran against.
  plan.tcp_path.reserve(plan.trajectory.q_a_ref.size());
  for (const crane_model::QA & q_a : plan.trajectory.q_a_ref) {
    auto settled = model.passive_equilibrium(q_a, request.payload);
    if (!settled.ok()) {
      return Result<MotionPlan>::failure(settled.status());
    }
    Q q = Q::Zero();
    for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
      q[static_cast<Eigen::Index>(kActuatedRows[row])] = q_a[static_cast<Eigen::Index>(row)];
    }
    q[4] = settled.value()[0];
    q[5] = settled.value()[1];
    auto pose = model.forward_kinematics(q, Frame::MountingBase, Frame::Tcp);
    if (!pose.ok()) {
      return Result<MotionPlan>::failure(pose.status());
    }
    plan.tcp_path.push_back(std::move(pose).value());
  }

  return Result<MotionPlan>::success(std::move(plan));
}

}  // namespace crane_planning
