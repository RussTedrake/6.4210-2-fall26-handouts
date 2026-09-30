"""Surface normals of a point cloud, estimated directly from the depth image."""

import sys
from pathlib import Path

# Add the repository root so utils can be imported.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.mustard_depth_camera_example import MustardExampleSystem
from matplotlib.patches import Rectangle
from pydrake.all import Rgba, RigidTransform, RotationMatrix

from utils.plotting import plt
from utils.viz import HEADLESS, get_meshcat, keep_meshcat_open

######################################################################
## Set up the scene
######################################################################

## Camera data for a mustard bottle on a table

# Replace infinite depth values with a distant background
BACKGROUND_DEPTH = 10.0

# Camera frustum depth in meters
FRUSTUM_DEPTH = 0.6

# Mustard bottle label.
MUSTARD_LABEL = 1

# Meshcat point-cloud crop
CROP_LOWER = [-0.3, -0.3, -0.3]
CROP_UPPER = [0.3, 0.3, 0.3]


def load_camera_data():
    """Return depth (meters), object mask, camera intrinsics, and pose X_WC."""
    diagram = MustardExampleSystem()
    context = diagram.CreateDefaultContext()

    # Copy the depth data so it outlives the Drake context. Remove its channel axis.
    depth = (
        diagram.GetOutputPort("camera0_depth_image").Eval(context).data.squeeze().copy()
    )
    depth[depth == np.inf] = BACKGROUND_DEPTH
    labels = diagram.GetOutputPort("camera0_label_image").Eval(context).data.squeeze()
    mask = labels == MUSTARD_LABEL

    camera = diagram.GetSubsystemByName("camera0")
    camera_context = camera.GetMyMutableContextFromRoot(context)
    X_WC = camera.body_pose_in_world_output_port().Eval(camera_context)
    cam_info = camera.default_depth_render_camera().core().intrinsics()

    meshcat = get_meshcat()
    if meshcat is not None:
        meshcat.SetProperty("/Background", "visible", False)
        cloud = diagram.GetOutputPort("camera0_point_cloud").Eval(context)
        meshcat.SetObject(
            "point_cloud", cloud.Crop(lower_xyz=CROP_LOWER, upper_xyz=CROP_UPPER)
        )
        show_depth_camera(cam_info, X_WC)
    return depth, mask, cam_info, X_WC


def show_depth_camera(cam_info, X_WC) -> None:
    """Draw camera0 and its view frustum in Meshcat."""
    meshcat = get_meshcat()
    if meshcat is None:
        return

    w, h = cam_info.width(), cam_info.height()
    corners = project_depth_to_pC(
        np.array(
            [
                [0.0, 0.0, FRUSTUM_DEPTH],
                [0.0, w - 1, FRUSTUM_DEPTH],
                [h - 1, w - 1, FRUSTUM_DEPTH],
                [h - 1, 0.0, FRUSTUM_DEPTH],
            ]
        ),
        cam_info,
    )
    # Four rays and four edges form the frustum
    starts = np.vstack([np.zeros((4, 3)), corners])
    ends = np.vstack([corners, np.roll(corners, -1, axis=0)])

    # X_WC moves the frustum and triad into the world frame
    meshcat.SetLineSegments(
        "depth_camera/frustum", starts.T, ends.T, 2.0, Rgba(0.1, 0.6, 1.0, 1.0)
    )
    AddMeshcatTriad(meshcat, "depth_camera/frame", length=0.05, radius=0.002)
    meshcat.SetTransform("depth_camera", X_WC)


def bbox(img: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """Half-open (row, column) bounds of a nonempty object mask."""
    v, u = np.nonzero(img)
    if len(v) == 0:
        raise ValueError("the object mask is empty")
    return (int(v.min()), int(v.max()) + 1), (int(u.min()), int(u.max()) + 1)


######################################################################
## Code for students to be aware of
######################################################################

## Convert every depth pixel (v, u, Z) to a camera-frame point


def project_depth_to_pC(depth_pixel: np.ndarray, cam_info) -> np.ndarray:
    """Project depth pixels into the camera frame.

    Input:
        depth_pixel: numpy array of (nx3), each row (v, u, Z)
    Output:
        pC: 3D points in the camera frame, numpy array of (nx3)
    """
    # Image arrays index (v, u)
    v = depth_pixel[:, 0]
    u = depth_pixel[:, 1]
    Z = depth_pixel[:, 2]
    # Camera intrinsics
    cx = cam_info.center_x()
    cy = cam_info.center_y()
    fx = cam_info.focal_x()
    fy = cam_info.focal_y()
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.column_stack([X, Y, Z])


def depth_points_in_camera_frame(depth: np.ndarray, cam_info) -> np.ndarray:
    """Convert an (H, W) depth image to (H, W, 3) camera-frame points."""
    v, u = np.indices(depth.shape)
    pixels = np.column_stack([v.ravel(), u.ravel(), depth.ravel()])
    return project_depth_to_pC(pixels, cam_info).reshape(*depth.shape, 3)


## Use 7x7 windows at 10-pixel intervals over the mask's bounding box.
## Keep these values fixed because they affect the graded result

WINDOW_HALF_WIDTH = 3
UV_STEP = 10


def scanning_windows(mask: np.ndarray, uv_step: int = UV_STEP):
    """Yield row and column slices for windows centered on object pixels."""
    if uv_step <= 0:
        raise ValueError("uv_step must be positive")
    if not np.any(mask):
        return
    height, width = mask.shape
    v_bounds, u_bounds = bbox(mask)
    radius = WINDOW_HALF_WIDTH
    for v in range(*v_bounds, uv_step):
        for u in range(*u_bounds, uv_step):
            if mask[v, u]:
                rows = slice(max(0, v - radius), min(height, v + radius + 1))
                cols = slice(max(0, u - radius), min(width, u + radius + 1))
                yield rows, cols


def normal_patches(pC: np.ndarray, mask: np.ndarray, uv_step: int = UV_STEP):
    """Yield each window's object points, keeping finite, positive depths.

    pC may have shape (H*W, 3) or (H, W, 3). Windows are visited in row-major order.
    """
    points = pC.reshape(*mask.shape, 3)
    for rows, cols in scanning_windows(mask, uv_step):
        patch = points[rows, cols][mask[rows, cols]]
        valid = np.all(np.isfinite(patch), axis=1) & (patch[:, 2] > 0)
        yield patch[valid]


######################################################################
## Code for students to write
######################################################################


def estimate_normal_by_nearest_pixels(
    X_WC: RigidTransform,
    pC: np.ndarray,
    mask: np.ndarray,
    uv_step: int = UV_STEP,
) -> list[RigidTransform]:
    """Return world poses of outward normal frames, in the order produced
    by a call to normal_patches(pC, mask, uv_step)."""
    raise NotImplementedError("your code here")


######################################################################
## Testing
######################################################################


def show_input():
    """Show the object's depth range, with black outside its mask."""
    depth, mask, _, _ = load_camera_data()
    _, ax = plt.subplots()
    depth = np.ma.masked_where(~mask, depth)
    depth = np.ma.masked_invalid(depth)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("black")
    # Only object depths determine the colormap's full dynamic range.
    image = ax.imshow(depth, cmap=cmap, interpolation="nearest")
    plt.colorbar(image, ax=ax, label="Depth (m)")
    ax.set_title("camera0 depth image")
    if not HEADLESS:
        plt.show()


def illustrate_window(uv_step=UV_STEP, margin=20):
    """Draw the estimator's masked window centers over the depth image."""
    depth, mask, _, _ = load_camera_data()
    v_bound, u_bound = bbox(mask)

    _, ax = plt.subplots()
    ax.imshow(depth)
    windows = 0
    for rows, cols in scanning_windows(mask, uv_step):
        # Matplotlib uses (x, y); pixel edges are half a pixel from their centers.
        ax.add_patch(
            Rectangle(
                (cols.start - 0.5, rows.start - 0.5),
                cols.stop - cols.start,
                rows.stop - rows.start,
                fill=False,
                edgecolor="r",
                linewidth=0.5,
            )
        )
        windows += 1
    ax.set_title(f"{windows} windows of {2 * WINDOW_HALF_WIDTH + 1} px, step {uv_step}")

    # The bottle is a fifth of the frame and a window is seven pixels across,
    # so at full frame these are specks.  Crop to the mask's box.  y is
    # inverted because an image's rows run downwards.
    ax.set_xlim(u_bound[0] - margin, u_bound[1] + margin)
    ax.set_ylim(v_bound[1] + margin, v_bound[0] - margin)
    if not HEADLESS:
        plt.show()


def test1():
    """Estimate the normals and draw a triad at each one in Meshcat"""
    depth, mask, cam_info, X_WC = load_camera_data()
    pC = depth_points_in_camera_frame(depth, cam_info)
    normals = estimate_normal_by_nearest_pixels(X_WC, pC, mask)
    print(f"{len(normals)} normal frames")
    meshcat = get_meshcat()
    if meshcat is not None:
        for i, X_WN in enumerate(normals):
            AddMeshcatTriad(
                meshcat, f"normal_vec_{i}", length=0.01, radius=0.001, X_PT=X_WN
            )


def test2():
    """Fixed planar patch for Gradescope; no renderer or Meshcat required."""
    x, y = np.meshgrid([-0.01, 0.0, 0.01], [-0.01, 0.0, 0.01])
    points = np.dstack([x, y, np.ones_like(x)])
    # One clipped window covers the whole small image.
    frames = estimate_normal_by_nearest_pixels(
        RigidTransform([0.1, -0.2, 0.3]), points, np.ones((3, 3), dtype=bool)
    )
    X_WN = frames[0]
    print(f"synthetic translation = {np.round(X_WN.translation(), 4)}")
    print(f"synthetic normal      = {np.round(X_WN.rotation().matrix()[:, 2], 4)}")


if __name__ == "__main__":
    show_input()
    # illustrate_window()
    # test1()
    # test2()
    keep_meshcat_open()
