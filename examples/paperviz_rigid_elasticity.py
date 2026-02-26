import argparse
import json
import math
import os

import imageio
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm, trange

from gradsim import dflex as df
from gradsim.renderutils import SoftRenderer, TriangleMesh
from gradsim.utils import meshutils
from gradsim.utils.logging import write_imglist_to_gif
from gradsim.utils.quaternion import quaternion_to_rotmat
try:
    from pxr import Usd, UsdGeom
except ImportError:
    pass


def write_meshes_to_file(vertices_across_time, faces, dirname):
    os.makedirs(dirname, exist_ok=True)
    for i, vertices in enumerate(vertices_across_time):
        np.savetxt(os.path.join(dirname, f"{i:03d}.txt"), vertices)
    np.savetxt(os.path.join(dirname, "faces.txt"), faces)


def get_world_vertices(vertices, quaternion, translation):
    """Returns vertices transformed to world-frame. """
    rotmat = quaternion_to_rotmat(quaternion, scalar_last=True)
    return jnp.matmul(rotmat, vertices.T).T + translation.reshape(1, 3)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for RNG."
    )
    parser.add_argument(
        "--expid",
        type=str,
        default="default",
        help="Unique string identifier for this experiment.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=os.path.join("cache", "rigid-elasticity"),
        help="Directory to store experiment logs in.",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default=os.path.join("sampledata", "lowpoly", "box.obj"),
        help="Path to input mesh file (.obj format).",
    )
    parser.add_argument(
        "--sim-duration",
        type=float,
        default=2.0,
        help="Duration of the simulation episode.",
    )
    parser.add_argument(
        "--physics-engine-rate",
        type=int,
        default=60,
        help="Number of physics engine `steps` per 1 second of simulator time.",
    )
    parser.add_argument(
        "--sim-substeps",
        type=int,
        default=16,
        help="Number of sub-steps to integrate, per 1 `step` of the simulation.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training iterations."
    )
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
    parser.add_argument(
        "--method",
        type=str,
        default="gradsim",
        choices=["noisy-physics-only", "physics-only", "gradsim"],
        help="Method to use, to optimize for initial velocity."
    )
    parser.add_argument(
        "--compare-every",
        type=int,
        default=1,
        help="Interval at which video frames are compared.",
    )
    parser.add_argument("--log", action="store_true", help="Log experiment data.")

    args = parser.parse_args()
    print(args)

    if args.log:
        print("LOGGING enabled!")
    else:
        print("LOGGING DISABLED!!")

    logdir = os.path.join(args.logdir, args.expid)
    if args.log:
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, "args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    sim_dt = (1.0 / args.physics_engine_rate) / args.sim_substeps
    sim_steps = int(args.sim_duration / sim_dt)
    sim_time = 0.0

    builder_gt = df.sim.ModelBuilder()
    obj = TriangleMesh.from_obj(args.mesh)
    vertices = meshutils.normalize_vertices(obj.vertices)
    points = np.array(vertices)
    indices = list(np.array(obj.faces).reshape((-1)))

    mesh = df.sim.Mesh(points, indices)

    pos_gt = (0.0, 4.0, 0.0)
    rot_gt = df.quat_from_axis_angle((0.0, 0.0, 1.0), math.pi * 0.3)
    vel_gt = (0.0, 2.0, 0.0)
    omega_gt = (0.0, 0.0, 0.0)
    scale_gt = (1.0, 1.0, 1.0)
    density_gt = 5.0
    ke_gt = 4900.0
    kd_gt = 15.0
    kf_gt = 990.0
    mu_gt = 0.77

    rigid_gt = builder_gt.add_rigid_body(
        pos=pos_gt, rot=rot_gt, vel=vel_gt, omega=omega_gt,
    )
    shape_gt = builder_gt.add_shape_mesh(
        rigid_gt,
        mesh=mesh,
        scale=scale_gt,
        density=density_gt,
        ke=ke_gt,
        kd=kd_gt,
        kf=kf_gt,
        mu=mu_gt,
    )
    model_gt = builder_gt.finalize("cpu")
    model_gt.ground = True

    integrator = df.sim.SemiImplicitIntegrator()

    renderer = SoftRenderer(camera_mode="look_at")
    camera_distance = 10.0
    elevation = 30.0
    azimuth = 0.0
    renderer.set_eye_from_angles(camera_distance, elevation, azimuth)

    render_every = 60 * 4

    vertices_jnp = jnp.array(points.astype(np.float32))
    faces_jnp = jnp.array(np.asarray(indices).reshape((-1, 3)))
    textures = jnp.concatenate(
        (
            jnp.ones((1, faces_jnp.shape[-2], 2, 1), dtype=jnp.float32),
            jnp.zeros((1, faces_jnp.shape[-2], 2, 1), dtype=jnp.float32),
            jnp.zeros((1, faces_jnp.shape[-2], 2, 1), dtype=jnp.float32),
        ),
        axis=-1,
    )

    imgs_gt = []
    positions_gt = []
    logvertices_gt = []
    faces_gt = None

    sim_time = 0.0
    state_gt = model_gt.state()

    model_gt.collide(state_gt)

    for i in trange(sim_steps):
        state_gt = integrator.forward(model_gt, state_gt, sim_dt)
        sim_time += sim_dt

        if i % render_every == 0 or i == sim_steps - 1:
            vertices_current = get_world_vertices(
                vertices_jnp, state_gt.rigid_r.reshape(-1), state_gt.rigid_x
            )
            rgba = renderer.forward(
                vertices_current[None, :],
                faces_jnp[None, :],
                textures,
            )
            imgs_gt.append(rgba)
            positions_gt.append(state_gt.rigid_x)
            logvertices_gt.append(np.array(vertices_current))

    if args.log:
        write_imglist_to_gif(imgs_gt, os.path.join(logdir, "gt.gif"))
        write_meshes_to_file(
            logvertices_gt,
            np.array(faces_jnp),
            os.path.join(logdir, "vertices_gt")
        )

    # Optimization section is commented out in this visualization script.
    # See paperviz_rigid_elasticity.py for the original torch-based optimization.
