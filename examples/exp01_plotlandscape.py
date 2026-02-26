"""
Plots the loss landscape for a mass estimation scenario.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from tqdm import trange

from gradsim.bodies import RigidBody
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
        default=os.path.join("cache", "mass_loss_landscape"),
        help="Directory to store logs in.",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed (for repeatability)"
    )
    parser.add_argument(
        "--infile",
        type=str,
        default=Path("sampledata/cube.obj"),
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
    body_gt = RigidBody(vertices[0], masses=masses_gt)

    gravity = ConstantForce(
        direction=jnp.array([0.0, -1.0, 0.0]),
        magnitude=args.force_magnitude,
    )

    body_gt.add_external_force(gravity, application_points=[0, 1])

    sim_gt = Simulator([body_gt])

    renderer = SoftRenderer(camera_mode="look_at")
    camera_distance = 8.0
    elevation = 30.0
    azimuth = 0.0
    renderer.set_eye_from_angles(camera_distance, elevation, azimuth)

    img_gt = []

    for i in trange(args.simsteps):
        sim_gt.step()
        rgba = renderer.forward(
            body_gt.get_world_vertices()[None, :], faces, textures
        )
        img_gt.append(rgba)

    mass_estimates = []
    losses = []
    mass_errors = []

    mass_interp = jnp.linspace(0.1, 5, 500, dtype=jnp.float32)

    for i in trange(mass_interp.size):
        masses_cur = mass_interp[i] * jnp.ones(vertices.shape[1], dtype=jnp.float32)
        body = RigidBody(vertices[0], masses=masses_cur)
        body.add_external_force(gravity, application_points=[0, 1])
        sim_est = Simulator([body])
        img_est = []
        for t in range(args.simsteps):
            sim_est.step()
            rgba = renderer.forward(
                body.get_world_vertices()[None, :], faces, textures
            )
            img_est.append(rgba)
        loss = sum(
            [
                jnp.mean((est - gt) ** 2)
                for est, gt in zip(
                    img_est[:: args.compare_every], img_gt[:: args.compare_every]
                )
            ]
        ) / (len(img_est[:: args.compare_every]))

        mass_estimates.append(float(mass_interp[i]))
        mass_errors.append(float(jnp.abs(masses_cur - masses_gt).mean()))
        losses.append(float(loss))

    if args.log:
        logdir = os.path.join(args.logdir, args.expid)
        os.makedirs(logdir, exist_ok=True)

        np.savetxt(os.path.join(logdir, "losses.txt"), losses)
        np.savetxt(os.path.join(logdir, "mass_estimates.txt"), mass_estimates)
        np.savetxt(os.path.join(logdir, "mass_errors.txt"), mass_errors)
