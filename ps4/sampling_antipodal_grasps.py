"""Sample antipodal grasps on a mesh and use one to rotate a letter."""

import sys
from pathlib import Path

# Add the repository root so utils can be imported.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from itertools import product

import numpy as np
import trimesh
from manipulation.letter_generation import create_sdf_asset_from_letter
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.station import LoadScenario, MakeHardwareStation
from pydrake.all import (
    AddFrameTriadIllustration,
    ConstantVectorSource,
    DiagramBuilder,
    Integrator,
    JacobianWrtVariable,
    PiecewisePolynomial,
    PiecewisePose,
    RigidTransform,
    RotationMatrix,
    Simulator,
    TrajectorySource,
)

from utils.dtsystems import DTSystem
from utils.viz import (
    HEADLESS,
    get_meshcat,
    keep_meshcat_open,
    publish_recording,
    start_recording,
)

######################################################################
## The scene: an iiwa, a table, and two letters lying flat on it
######################################################################


# Letter geometry and contact parameters.
LETTER_HEIGHT = 0.25
LETTER_DEPTH = 0.07
LETTER_MASS = 0.1
LETTER_MU_STATIC = 1.17

ASSET_DIR = Path(__file__).parent.resolve() / "assets"

# Initial letter poses; the first is rotated 30 degrees.
LETTER_LAYOUT = [(-0.2, 0.0, 30), (0.25, 0.0, 0)]


def write_letter_assets() -> None:
    """Generate one SDF and OBJ per distinct initial, under assets/."""
    for letter in dict.fromkeys(INITIALS):
        create_sdf_asset_from_letter(
            text=letter,
            font_name="DejaVu Sans",
            letter_height_meters=LETTER_HEIGHT,
            extrusion_depth_meters=LETTER_DEPTH,
            output_dir=str(ASSET_DIR),
            include_normals=True,
            mu_static=LETTER_MU_STATIC,
            mass=LETTER_MASS,
        )


def letter_model(i: int) -> str:
    """Use a unique model name even when both initials are the same."""
    return f"{INITIALS[i]}_{i}"


def letter_body_name(i: int) -> str:
    return f"{INITIALS[i]}_body_link"


def letter_directive(i: int, x: float, y: float, yaw: float) -> str:
    return f"""
- add_model:
    name: {letter_model(i)}
    file: file://{ASSET_DIR}/{INITIALS[i]}.sdf
    default_free_body_pose:
        {letter_body_name(i)}:
            translation: [{x}, {y}, 0]
            rotation: !Rpy {{ deg: [0, 0, {yaw}] }}"""


def scenario_yaml() -> str:
    letters = "".join(
        letter_directive(i, *place) for i, place in enumerate(LETTER_LAYOUT)
    )
    return f"""directives:
- add_model:
    name: iiwa
    file: package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf
    default_joint_positions:
        iiwa_joint_1: [-1.57]
        iiwa_joint_2: [0.1]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.2]
        iiwa_joint_5: [0]
        iiwa_joint_6: [1.6]
        iiwa_joint_7: [0]
- add_weld:
    parent: world
    child: iiwa::iiwa_link_0
    X_PC:
        translation: [0, -0.5, 0]
        rotation: !Rpy {{ deg: [0, 0, 180] }}
- add_model:
    name: wsg
    file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
        translation: [0, 0, 0.09]
        rotation: !Rpy {{ deg: [90, 0, 90] }}
- add_model:
    name: table
    file: package://manipulation/table.sdf
- add_weld:
    parent: world
    child: table::table_link
    X_PC:
        translation: [0.0, 0.0, -0.05]
        rotation: !Rpy {{ deg: [0, 0, -90] }}{letters}

model_drivers:
    iiwa: !IiwaDriver
        control_mode: position_only
        hand_model_name: wsg
    wsg: !SchunkWsgDriver {{}}
"""


def load_letter_mesh(letter: str) -> trimesh.Trimesh:
    """Load the letter as one object-frame mesh."""
    return trimesh.load(ASSET_DIR / f"{letter}.obj", force="mesh")


######################################################################
## Code for students to be aware of
######################################################################

## Pseudo-inverse controller from ps2: gripper velocity to joint velocity.


def arm_jacobian_columns(plant) -> slice:
    """The columns of the plant's Jacobian that belong to the arm.

    The plant holds the letters as well as the robot, so a gripper Jacobian
    comes back 6 x N; only these seven columns are joints we can command.
    """
    return slice(
        plant.GetJointByName("iiwa_joint_1").velocity_start(),
        plant.GetJointByName("iiwa_joint_7").velocity_start() + 1,
    )


def JointVelocity(plant):
    """Which joint velocities produce a wanted gripper velocity.

    Input ports: V_WG, iiwa.position
    Output: iiwa.velocity
    """
    plant_context = plant.CreateDefaultContext()
    iiwa = plant.GetModelInstanceByName("iiwa")
    G = plant.GetBodyByName("body", plant.GetModelInstanceByName("wsg")).body_frame()
    W = plant.world_frame()
    arm_columns = arm_jacobian_columns(plant)

    def solve(_state, inputs):
        # Keep the arm's seven Jacobian columns, then solve J q_dot = V_WG.
        V_WG, q = inputs
        plant.SetPositions(plant_context, iiwa, q)
        J_G = plant.CalcJacobianSpatialVelocity(
            plant_context, JacobianWrtVariable.kV, G, [0, 0, 0], W, W
        )
        return np.linalg.pinv(J_G[:, arm_columns]) @ V_WG

    return DTSystem(
        [6, 7],
        0,
        7,
        None,
        solve,
        output_depends_on_input=True,
        input_port_name=["V_WG", "iiwa.position"],
        output_port_name="iiwa.velocity",
        name="JointVelocity",
    )


# Candidate tuple: (point_1, point_2, normal_1, normal_2), in frame O.
AntipodeCandidateType = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

# Offset from the grasp point to the gripper body frame.
FINGER_LENGTH = 0.10

# A coarse gripper-body box in G: test all eight corners after rotation.
GRIPPER_BOUNDS = np.array([[-0.073, -0.085383, -0.025], [0.073, 0.069, 0.025]])
GRIPPER_VERTICES = np.array(list(product(*GRIPPER_BOUNDS.T)))


def compute_prepick_pose(X_WG: RigidTransform) -> RigidTransform:
    """Back off along the gripper approach axis."""
    return X_WG @ RigidTransform([0, -0.17, 0.0])


# How far off the table to carry the letter once the fingers have closed.
LIFT_HEIGHT = 0.15


def lift(X_WG: RigidTransform, height: float = LIFT_HEIGHT) -> RigidTransform:
    """Raise a gripper pose straight up in the *world*, keeping its rotation.

    Pre-multiplying is what makes the offset a world one: post-multiplying
    would move along the gripper's own axes, and the gripper is pointing down.
    """
    return RigidTransform([0.0, 0.0, height]) @ X_WG


def build_station():
    """The scene on its own, with no controller in it."""
    write_letter_assets()
    builder = DiagramBuilder()
    station = MakeHardwareStation(
        LoadScenario(data=scenario_yaml()), meshcat=get_meshcat()
    )
    builder.AddSystem(station)
    return builder, station


def letter_pose(plant, plant_context, i: int) -> RigidTransform:
    """Where the i'th initial has got to."""
    model = plant.GetModelInstanceByName(letter_model(i))
    return plant.EvalBodyPoseInWorld(
        plant_context, plant.GetBodyByName(letter_body_name(i), model)
    )


def add_triads(station, plant) -> None:
    scene_graph = station.GetSubsystemByName("scene_graph")
    for i in range(len(INITIALS)):
        model = plant.GetModelInstanceByName(letter_model(i))
        AddFrameTriadIllustration(
            scene_graph=scene_graph,
            body=plant.GetBodyByName(letter_body_name(i), model),
            length=0.1,
        )
    AddFrameTriadIllustration(
        scene_graph=scene_graph, body=plant.GetBodyByName("body"), length=0.1
    )


def hold_arm_position(builder, plant, station):
    """Command the arm to stay at its initial joint positions."""
    iiwa = plant.GetModelInstanceByName("iiwa")
    hold_arm = builder.AddSystem(ConstantVectorSource(plant.GetDefaultPositions(iiwa)))
    builder.Connect(hold_arm.get_output_port(), station.GetInputPort("iiwa.position"))


def start_the_fingers_open(plant):
    """Have the fingers already be apart when the simulation starts."""
    plant.SetDefaultPositions(
        plant.GetModelInstanceByName("wsg"), [-OPENED / 2, OPENED / 2]
    )


def keep_the_fingers_open(builder, plant, station):
    """Start the fingers open, and keep commanding them open."""
    start_the_fingers_open(plant)
    command = builder.AddSystem(ConstantVectorSource([OPENED]))
    builder.Connect(command.get_output_port(), station.GetInputPort("wsg.position"))


# Draw a small subset so the letter stays visible.
GRASPS_TO_DRAW = 10


def draw_grasps(grasps: list[RigidTransform], n: int = GRASPS_TO_DRAW) -> None:
    """Draw an evenly spaced handful of `grasps` in the world, as triads."""
    meshcat = get_meshcat()
    if meshcat is None or not grasps:
        return
    for i in np.unique(np.linspace(0, len(grasps) - 1, n).astype(int)):
        AddMeshcatTriad(
            meshcat, path=f"grasps/{i}", X_PT=grasps[i], length=0.05, opacity=0.5
        )


def build_static_system():
    """The scene, with the arm parked at the scenario's own pose."""
    builder, station = build_station()
    plant = station.GetSubsystemByName("plant")
    add_triads(station, plant)
    keep_the_fingers_open(builder, plant, station)
    hold_arm_position(builder, plant, station)
    return builder.Build()


# What the integrator is called in the diagram, so that show_scene can find it
# again to start it off.
JOINT_INTEGRATOR = "JointIntegrator"


def start_the_integrator(system, context):
    """Initialize the controller's integrator with the arm's starting angles.

    Static scenes have no integrator and need no initialization.
    """
    integrator = next(
        (s for s in system.GetSystems() if s.get_name() == JOINT_INTEGRATOR), None
    )
    if integrator is None:
        return
    plant = system.GetSubsystemByName("station").GetSubsystemByName("plant")
    integrator.set_integral_value(
        integrator.GetMyContextFromRoot(context),
        plant.GetDefaultPositions(plant.GetModelInstanceByName("iiwa")),
    )


def show_scene(system, duration: float = 2.0):
    """Simulate a system in real time, so it can be watched in Meshcat."""
    simulator = Simulator(system)
    context = simulator.get_mutable_context()
    start_the_integrator(system, context)
    system.ForcedPublish(context)

    start_recording()
    if not HEADLESS:
        simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(duration)
    publish_recording()
    return simulator


######################################################################
## Code for students to study
######################################################################


# How long every leg of a plan is given, whatever it has to do in it.
KEYFRAME_SECONDS = 3.0


def build_system_tracking_keyframes(keyframes):
    """Build the tracking diagram and return its duration."""
    builder, station = build_station()
    plant = station.GetSubsystemByName("plant")
    start_the_fingers_open(plant)

    # Repeat the last keyframe to finish with a stationary segment.
    keyframes = list(keyframes) + [keyframes[-1]]

    sample_times = [KEYFRAME_SECONDS * i for i in range(len(keyframes))]
    traj_X_G = PiecewisePose.MakeLinear(sample_times, [kf[0] for kf in keyframes])
    # Differentiate poses to get spatial velocity.
    traj_V_G = traj_X_G.MakeDerivative()
    traj_wsg = PiecewisePolynomial.FirstOrderHold(
        sample_times, np.array([kf[1] for kf in keyframes])[None]
    )

    V_G_source = builder.AddSystem(TrajectorySource(traj_V_G))
    controller = builder.AddSystem(JointVelocity(plant))
    integrator = builder.AddSystem(Integrator(7))
    integrator.set_name(JOINT_INTEGRATOR)
    wsg_source = builder.AddSystem(TrajectorySource(traj_wsg))

    builder.Connect(V_G_source.get_output_port(), controller.GetInputPort("V_WG"))
    builder.Connect(controller.get_output_port(), integrator.get_input_port())
    builder.Connect(integrator.get_output_port(), station.GetInputPort("iiwa.position"))
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        controller.GetInputPort("iiwa.position"),
    )
    builder.Connect(wsg_source.get_output_port(), station.GetInputPort("wsg.position"))

    add_triads(station, plant)
    return builder.Build(), traj_V_G.end_time()


## Rotate 30 degrees about the gripper approach axis.
TURN_ANGLE = np.pi / 6

# Open wider than the 0.04 m maximum sampled contact separation.
OPENED = 0.05
CLOSED = 0.0


def check_collision_free(X_WG: RigidTransform) -> bool:
    """Coarse table-only test: all eight body-box corners must have z_W > 0.

    Does not test fingers, the object mesh, the robot arm, or approach paths.
    """
    verts_W = (X_WG @ GRIPPER_VERTICES.T).T
    return bool(np.all(verts_W[:, 2] > 0.0))


######################################################################
## Code for students to write
######################################################################

INITIALS = ["B", "B"]  # Your initials!


def plan(X_WGinitial: RigidTransform, X_WGpick: RigidTransform):
    """Return ten (X_WG, opening) keyframes for approach, pick, and placement.

    Use compute_prepick_pose and lift. Rotate by TURN_ANGLE about +G_y at
    the pinch point, keeping that point fixed during the turn. Close/open
    while the gripper is stationary; retreat before returning to the start.
    """
    raise NotImplementedError("your code here")


def sample_colinear_points(
    mesh: trimesh.Trimesh, n_sample_points: int, rng: np.random.Generator | None = None
) -> list[AntipodeCandidateType]:
    """Return (p1, p2, n1, n2) tuples in object coordinates; normals are outward.

    Sample n_sample_points with trimesh.sample.sample_surface. Cast each ray
    from p1 - 1e-6*n1 along -n1; discard hits <= 1e-6 m from p1 and hits
    behind the origin, then choose the nearest remaining hit. Return [] if
    none survive. Use the hit face index to obtain n2. Pass rng as the
    sample_surface seed so demonstrations are repeatable.
    """
    raise NotImplementedError("your code here")


def compute_grasp_from_points(
    antipodal_pt: AntipodeCandidateType,
) -> RigidTransform | None:
    """Return X_OG, or None for coincident points or a vertical chord.

    Align +G_x with p1-p2; project -O_z perpendicular to it for +G_y.
    Set +G_z = G_x cross G_y and put the fingertip midpoint at (p1+p2)/2.
    The body origin is FINGER_LENGTH behind that midpoint along G_y.
    """
    raise NotImplementedError("your code here")


def get_filtered_grasps(
    candidate_list: list[AntipodeCandidateType],
    z_axis_thresh: float,
    max_pt_dist: float,
    min_pt_dist: float,
    X_WO: RigidTransform,
    mu: float = LETTER_MU_STATIC,
) -> list[RigidTransform]:
    """Return surviving X_WG poses in input order, for letters lying flat.

    Both inward normals must make angle <= atan(mu) with the chord pointing
    toward the other contact. Also require min_pt_dist <= width <= max_pt_dist
    and abs(n1.O_z) <= z_axis_thresh. Transform by X_WO, then apply the given
    coarse collision check. Normals are unit length.
    """
    raise NotImplementedError("your code here")


def grasp_moment_arm(X_WG: RigidTransform, p_Wcom: np.ndarray) -> float:
    """Return the horizontal pinch-to-COM distance in meters.

    X_WG is the gripper body pose, and p_Wcom is the world COM position.
    Locate the pinch at [0, FINGER_LENGTH, 0] in G before comparing positions.
    """
    raise NotImplementedError("your code here")


def sample_grasp(
    mesh: trimesh.Trimesh,
    X_WO: RigidTransform,
    n_sample_pts: int = 500,
    rng: np.random.Generator | None = None,
) -> RigidTransform:
    """One grasp on the object at X_WO.  Raises if the sampling found none."""
    colinear_pts = sample_colinear_points(mesh, n_sample_points=n_sample_pts, rng=rng)
    candidate_grasps = get_filtered_grasps(
        colinear_pts,
        z_axis_thresh=0.8,
        max_pt_dist=0.04,
        min_pt_dist=0.005,
        X_WO=X_WO,
    )
    if not candidate_grasps:
        raise RuntimeError(
            f"no grasp survived filtering out of {len(colinear_pts)} candidates; "
            "try more sample points or looser thresholds"
        )
    # Return the one most nearly over the center of mass
    p_Wcom = X_WO @ mesh.center_mass
    return min(candidate_grasps, key=lambda X_WG: grasp_moment_arm(X_WG, p_Wcom))


######################################################################
## Testing
######################################################################


def test1():
    """Show the scene"""
    show_scene(build_static_system())


def test2(seed=0):
    """Sample and report grasps for the first letter"""
    rng = np.random.default_rng(seed)
    system = build_static_system()
    plant = system.GetSubsystemByName("station").GetSubsystemByName("plant")
    context = system.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    X_WO = letter_pose(plant, plant_context, 0)
    mesh = load_letter_mesh(INITIALS[0])

    colinear = sample_colinear_points(mesh, 500, rng=rng)
    grasps = get_filtered_grasps(
        colinear,
        z_axis_thresh=0.8,
        max_pt_dist=0.04,
        min_pt_dist=0.005,
        X_WO=X_WO,
    )
    print(f"{len(colinear)} colinear pairs -> {len(grasps)} grasps")
    draw_grasps(grasps)
    show_scene(system)


def test3():
    """Fixed candidate for Gradescope, without assets or simulation."""
    p1 = np.array([-0.01, 0.0, 0.05])
    p2 = np.array([0.01, 0.0, 0.05])
    candidate = (p1, p2, np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    X_WG = compute_grasp_from_points(candidate)  # O and W coincide
    p_Wcom = np.array([0.03, 0.04, 0.05])
    print(f"width = {np.linalg.norm(p2 - p1):.4f}")
    print(f"moment arm = {grasp_moment_arm(X_WG, p_Wcom):.4f}")
    print(f"table clearance = {check_collision_free(X_WG)}")


def test4():
    """Check that plan turns about the pinch point, without simulation."""
    candidate = (
        np.array([-0.01, 0.0, 0.05]),
        np.array([0.01, 0.0, 0.05]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    X_WG = compute_grasp_from_points(candidate)  # O and W coincide
    keyframes = plan(X_WG, X_WG)
    p_GP = np.array([0.0, FINGER_LENGTH, 0.0])
    before, after = keyframes[4][0], keyframes[5][0]
    print(f"pinch displacement = {np.round(after @ p_GP - before @ p_GP, 4)}")
    R_delta = after.rotation().matrix() @ before.rotation().matrix().T
    print(
        f"world yaw change = {np.degrees(np.arctan2(R_delta[1, 0], R_delta[0, 0])):.4f}"
    )


def test5(seed=0):
    """Pick and rotate the first letter to match the second"""
    rng = np.random.default_rng(seed)
    _, station = build_station()
    plant = station.GetSubsystemByName("plant")
    context = station.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    X_WGinitial = plant.EvalBodyPoseInWorld(plant_context, plant.GetBodyByName("body"))
    X_WO = letter_pose(plant, plant_context, 0)
    X_WGpick = sample_grasp(load_letter_mesh(INITIALS[0]), X_WO, rng=rng)

    system, duration = build_system_tracking_keyframes(plan(X_WGinitial, X_WGpick))
    print(f"sanity check, simulation will run for {duration} seconds")
    simulator = show_scene(system, duration + 2.0)
    final_plant = system.GetSubsystemByName("station").GetSubsystemByName("plant")
    final_context = final_plant.GetMyContextFromRoot(simulator.get_context())
    actual = letter_pose(final_plant, final_context, 0).rotation()
    target = letter_pose(final_plant, final_context, 1).rotation()
    error = (target.inverse() @ actual).ToAngleAxis().angle()
    print(f"final letter orientation error = {np.degrees(error):.3f} degrees")


if __name__ == "__main__":
    test1()
    # test2()
    # test3()
    # test4()
    # test5()
    keep_meshcat_open()
