import argparse
import math
import os

import numpy as np
import jax
import jax.numpy as jnp
from tqdm import trange

from gradsim.bodies import RigidBody
from gradsim.engines import SemiImplicitEulerWithContacts
from gradsim.forces import ConstantForce
from gradsim.renderutils import SoftRenderer, TriangleMesh
from gradsim.simulator import Simulator
from gradsim.utils import meshutils

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expid",
        type=str,
        default="default",
        help="Unique string identifier for experiments.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=os.path.join("cache", "exp2_loss_landscape"),
        help="Directory to store logs in.",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed (for repeatability)"
    )
    parser.add_argument(
        "--infile",
        type=str,
        default=os.path.join("sampledata", "cube.obj"),
        help="Path to input mesh (.obj) file.",
    )
    parser.add_argument(
        "--simsteps",
        type=int,
        default=20,
        help="Number of steps to run simulation for.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to run optimization for.",
    )
    parser.add_argument(
        "--compare-every",
        type=int,
        default=1,
        help="Apply loss every `--compare-every` frames.",
    )
    parser.add_argument(
        "--uniform-density",
        action="store_true",
        help="Whether to treat the object as having uniform density.",
    )
    parser.add_argument(
        "--force-magnitude",
        type=float,
        default=10.0,
        help="Magnitude of external force.",
    )
    parser.add_argument("--log", action="store_true", help="Save log files.")

    args = parser.parse_args()

    if args.compare_every >= args.simsteps:
        raise ValueError(
            f"Arg --compare-every cannot be greater than or equal to {args.simsteps}."
        )

    mesh = TriangleMesh.from_obj(args.infile)
    vertices = meshutils.normalize_vertices(mesh.vertices[None, :])
    faces = mesh.faces[None, :]
    textures = jnp.concatenate(
        (
            jnp.ones((1, faces.shape[1], 2, 1), dtype=jnp.float32),
            jnp.ones((1, faces.shape[1], 2, 1), dtype=jnp.float32),
            jnp.zeros((1, faces.shape[1], 2, 1), dtype=jnp.float32),
        ),
        axis=-1,
    )
    masses_gt = jnp.ones(vertices.shape[1], dtype=jnp.float32)
    position = jnp.array([0.0, 4.0, 0.0], dtype=jnp.float32)
    orientation = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    restitution_gt = 0.8
    body = RigidBody(
        vertices[0],
        position=position,
        orientation=orientation,
        masses=masses_gt,
        restitution=restitution_gt,
    )

    force_magnitude = 10.0
    gravity = ConstantForce(
        magnitude=force_magnitude,
        direction=jnp.array([0.0, -1.0, 0.0]),
    )

    body.add_external_force(gravity)

    sim_duration = 1.5
    fps = 30
    sim_substeps = 32
    dtime = (1 / 30) / sim_substeps
    sim_steps = int(sim_duration / dtime)
    render_every = sim_substeps

    sim_gt = Simulator(
        bodies=[body], engine=SemiImplicitEulerWithContacts(), dtime=dtime
    )

    renderer = SoftRenderer(camera_mode="look_at")
    camera_distance = 10.0
    elevation = 30.0
    azimuth = 0.0
    renderer.set_eye_from_angles(camera_distance, elevation, azimuth)

    imgs_gt = []
    for i in trange(sim_steps):
        sim_gt.step()
        if i % render_every == 0:
            rgba = renderer.forward(
                body.get_world_vertices()[None, :], faces, textures
            )
            imgs_gt.append(rgba)

    mass_estimates = []
    mass_errors = []
    restitution_estimates = []
    restitution_errors = []
    losses = []

    mass_interp = jnp.linspace(0.1, 5, 50, dtype=jnp.float32)
    e_interp = jnp.linspace(0.4, 1.0, 10, dtype=jnp.float32)

    for i in trange(mass_interp.size):
        for j in trange(e_interp.size):
            masses_cur = mass_interp[i] * jnp.ones(vertices.shape[1], dtype=jnp.float32)
            e_cur = float(e_interp[j])
            body_est = RigidBody(vertices[0], masses=masses_cur, restitution=e_cur)
            body_est.add_external_force(gravity)
            sim_est = Simulator(
                bodies=[body_est], engine=SemiImplicitEulerWithContacts(), dtime=dtime
            )

            imgs_est = []
            for t in range(sim_steps):
                sim_est.step()
                if t % render_every == 0:
                    rgba = renderer.forward(
                        body_est.get_world_vertices()[None, :], faces, textures
                    )
                    imgs_est.append(rgba)
            loss = sum(
                [
                    jnp.mean((est - gt) ** 2)
                    for est, gt in zip(
                        imgs_est[:: args.compare_every], imgs_gt[:: args.compare_every]
                    )
                ]
            ) / (len(imgs_est[:: args.compare_every]))

            mass_estimates.append(float(mass_interp[i]))
            mass_errors.append(float(jnp.abs(masses_cur - masses_gt).mean()))
            restitution_estimates.append(e_cur)
            restitution_errors.append(abs(restitution_gt - e_cur))
            losses.append(float(loss))

    if args.log:
        logdir = os.path.join(args.logdir, args.expid)
        os.makedirs(logdir, exist_ok=True)

        np.savetxt(os.path.join(logdir, "losses.txt"), losses)
        np.savetxt(os.path.join(logdir, "mass_estimates.txt"), mass_estimates)
        np.savetxt(os.path.join(logdir, "mass_errors.txt"), mass_errors)
        np.savetxt(
            os.path.join(logdir, "restitution_estimates.txt"), restitution_estimates
        )
        np.savetxt(os.path.join(logdir, "restitution_errors.txt"), restitution_errors)
