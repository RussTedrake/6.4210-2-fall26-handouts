"""Find antipodal points on a parametric shape analytically."""

import sys
from pathlib import Path

# Add the repository root so utils can be imported.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from collections.abc import Callable

import numpy as np
from pydrake.all import (
    Evaluate,
    Jacobian,
    MathematicalProgram,
    Solve,
    Variable,
    atan,
    cos,
    eq,
    sin,
)

from utils.plotting import plt
from utils.viz import HEADLESS

######################################################################
## Code for students to be aware of
######################################################################

## Drake's trig functions support both floats and symbolic Variables.

KAPPA = 2.0


def shape(t: float) -> np.ndarray:
    """p(t): a point on the gear, for t in [0, 2 pi)."""
    x = (
        (10 * cos(t))
        - (1.5 * cos(t + atan(sin(-9 * t) / ((4 / 3) - cos(-9 * t)))))
        - (0.75 * cos(10 * t))
    )
    y = (
        (-10 * sin(t))
        + (1.5 * sin(t + atan(sin(-9 * t) / ((4 / 3) - cos(-9 * t)))))
        + (0.75 * sin(10 * t))
    )
    return np.array([x, y])


def symbolic_derivatives_demo() -> None:
    """How Drake's symbolic differentiation works, on T = cos^2(x) + y^5.

    Expected:
        J = [-2 cos(x) sin(x),  5 y^4]
        H = [[2 sin^2(x) - 2 cos^2(x), 0], [0, 20 y^3]]
    """
    x = Variable("x")
    y = Variable("y")

    # Build an expression from symbolic variables.
    T = cos(x) ** 2.0 + y**5.0
    print(f"T = {T}")

    # Use the method for scalars and Evaluate for arrays.
    print(f"T(3, 5) = {T.Evaluate({x: 3.0, y: 5.0})}")

    # These remain symbolic until evaluated.
    J = T.Jacobian([x, y])
    print(f"J = {J}")
    print(f"J(3, 5) = {Evaluate(J, {x: 3.0, y: 5.0})}")

    H = Jacobian(J, [x, y])
    print(f"H = {H}")
    print(f"H(3, 5) = {Evaluate(H, {x: 3.0, y: 5.0})}")


def plot_gear() -> None:
    theta = np.linspace(0, 2 * np.pi, 500)
    gear_shape = np.array([Evaluate(shape(t)).squeeze() for t in theta])
    plt.axis("equal")
    plt.plot(gear_shape[:, 0], gear_shape[:, 1], "k-")


def plot_antipodal_pts(pts, shape: Callable) -> None:
    antipodal_pts = np.array([Evaluate(shape(pts[i])).squeeze() for i in range(2)])
    plt.scatter(antipodal_pts[:, 0], antipodal_pts[:, 1], color="red")


def is_antipodal_pair(shape, parameters, orientation=-1, tolerance=1e-5):
    """Check distinct, regular, inward-facing normal contacts on a simple curve.

    orientation is +1 for counterclockwise or -1 for clockwise traversal.
    This is a local contact check, not a collision or reachability test.
    """
    t = Variable("boundary_parameter")
    p = shape(t)
    tangent = Jacobian(p, [t]).reshape(2)
    points = [Evaluate(p, {t: float(value)}).reshape(2) for value in parameters]
    tangents = [Evaluate(tangent, {t: float(value)}).reshape(2) for value in parameters]
    chord = points[1] - points[0]
    length = np.linalg.norm(chord)
    if length <= 1e-4:
        return False
    direction = chord / length
    for velocity, inward_chord in zip(tangents, [direction, -direction]):
        speed = np.linalg.norm(velocity)
        if speed <= 1e-8:
            return False
        tangent_hat = velocity / speed
        inward_normal = orientation * np.array([-tangent_hat[1], tangent_hat[0]])
        if abs(tangent_hat.dot(inward_chord)) > tolerance:
            return False
        if inward_normal.dot(inward_chord) <= 0:
            return False
    return True


######################################################################
## Code for students to write
######################################################################


def find_antipodal_pts(
    shape: Callable,
    rng: np.random.Generator | None = None,
    max_attempts: int = 100,
    orientation: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Find a normal-aligned antipodal pair and sorted Hessian eigenvalues.

    Solve grad(E)=0 for E=(KAPPA/2)*||p(t1)-p(t2)||^2, with no objective.
    Require 0 <= ti <= 2*pi-eps and t1-t2 >= eps, eps=1e-3.
    Try at most max_attempts uniform random guesses using rng. After solver
    success check the bounds (tolerance 1e-6), gradient (infinity norm <=
    1e-5), and is_antipodal_pair. Raise RuntimeError if all attempts fail.
    orientation describes the simple closed curve's traversal (+1 CCW, -1 CW).
    """
    raise NotImplementedError("your code here")


######################################################################
## Testing
######################################################################


def test1():
    """The symbolic-differentiation warm-up."""
    symbolic_derivatives_demo()


def test2():
    """Find a pair of antipodal points and draw them on the gear."""
    plot_gear()
    result, H_eig = find_antipodal_pts(shape)
    plot_antipodal_pts(result, shape)
    print(f"t  = {np.round(result, 4)}")
    print(f"H_eig = {np.round(H_eig, 4)}")
    if not HEADLESS:
        plt.show()


def test3():
    """Exact stationary ellipse pairs: stable numeric targets for Gradescope."""
    t = np.array([Variable("t1"), Variable("t2")])

    def ellipse(angle):
        return np.array([2 * cos(angle), sin(angle)])

    chord = ellipse(t[0]) - ellipse(t[1])
    E = 0.5 * KAPPA * chord.dot(chord)
    H = Jacobian(E.Jacobian(t), t)
    for name, parameters in [
        ("major", [0.0, np.pi]),
        ("minor", [np.pi / 2, 3 * np.pi / 2]),
    ]:
        eigenvalues = np.linalg.eigvalsh(Evaluate(H, dict(zip(t, parameters))))
        print(f"{name}-axis Hessian eigenvalues = {np.round(eigenvalues, 4)}")


def test4():
    """Look for three signatures; report missing cases instead of looping forever."""
    rng = np.random.default_rng(45)
    found = {}
    for _ in range(100):
        try:
            parameters, eigenvalues = find_antipodal_pts(shape, rng, max_attempts=5)
        except RuntimeError:
            continue
        if np.all(eigenvalues > 1e-6):
            label = "Minimum"
        elif np.all(eigenvalues < -1e-6):
            label = "Maximum"
        elif eigenvalues[0] < -1e-6 and eigenvalues[1] > 1e-6:
            label = "Saddle"
        else:
            label = "Inconclusive"
        found.setdefault(label, (parameters, eigenvalues))
        if {"Minimum", "Maximum", "Saddle"} <= found.keys():
            break
    for i, label in enumerate(["Minimum", "Maximum", "Saddle"]):
        plt.subplot(1, 3, i + 1)
        plot_gear()
        plt.title(label)
        if label in found:
            parameters, eigenvalues = found[label]
            plot_antipodal_pts(parameters, shape)
            print(
                f"{label}: t={np.round(parameters, 4)}, eigenvalues={np.round(eigenvalues, 4)}"
            )
        else:
            print(f"{label}: not found within the search budget")
    if not HEADLESS:
        plt.show()


if __name__ == "__main__":
    test1()
    # test2()
    # test3()
    # test4()
