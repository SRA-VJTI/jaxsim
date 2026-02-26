"""
Optimize parameters of a bouncing ball in 2D.
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
from tqdm import tqdm, trange

from gradsim.renderutils import SoftRenderer, TriangleMesh
from gradsim.utils import meshutils
from gradsim.utils.logging import write_imglist_to_gif


def _adam_init(params):
    return {k: {"m": jnp.zeros_like(v), "v": jnp.zeros_like(v), "t": 0}
            for k, v in params.items()}


def _adam_step(params, grads, state, lr, beta1=0.5, beta2=0.99, eps=1e-8):
    new_params, new_state = {}, {}
    for k in params:
        t = state[k]["t"] + 1
        m = beta1 * state[k]["m"] + (1 - beta1) * grads[k]
        v = beta2 * state[k]["v"] + (1 - beta2) * grads[k] ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        new_params[k] = params[k] - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        new_state[k] = {"m": m, "v": v, "t": t}
    return new_params, new_state


class BouncingBall2D:
    def __init__(
        self,
        pos=None,
        radius=1.0,
        theta=0.0,
        height=1.0,
        speed=1.0,
        gravity=-10.0,
    ):
        self.radius = jnp.array([radius], dtype=jnp.float32) if not hasattr(radius, 'shape') else radius
        self.theta = jnp.array([theta], dtype=jnp.float32) if not hasattr(theta, 'shape') else theta
        self.height = jnp.array([height], dtype=jnp.float32) if not hasattr(height, 'shape') else height
        self.speed = jnp.array([speed], dtype=jnp.float32) if not hasattr(speed, 'shape') else speed

        if pos is None:
            self.position_initial = jnp.zeros(2, dtype=jnp.float32)
        else:
            self.position_initial = pos

        self.velocity_initial = jnp.zeros(2, dtype=jnp.float32)
        self.position_initial = self.position_initial.at[1].set(self.height[0])
        self.velocity_initial = self.velocity_initial.at[0].set(self.speed[0] * jnp.cos(self.theta[0]))
        self.velocity_initial = self.velocity_initial.at[1].set(self.speed[0] * jnp.sin(self.theta[0]))

        self.gravity = jnp.array([0.0, gravity], dtype=jnp.float32)

        self.position_cur = self.position_initial
        self.velocity_cur = self.velocity_initial

        self.eps = 1e-16

    def step(self, dtime):
        vel_cache = self.velocity_cur
        pos_cache = self.position_cur

        # Leapfrog method
        pos = pos_cache + vel_cache * dtime / 2
        vel = vel_cache + self.gravity * dtime
        pos = pos + vel * dtime / 2

        # Collision handling with jnp.where (avoids Python conditionals on abstract values)
        collides = pos[1] < self.radius[0]
        dtime_new = (self.radius[0] - pos_cache[1]) / (vel_cache[1] + self.eps)
        vel_after = jnp.array([vel[0], -(vel_cache[1] + self.gravity[1] * dtime_new)])
        pos_after = jnp.array([pos_cache[0] + vel_cache[0] * dtime_new, self.radius[0]])

        self.position_cur = jnp.where(collides, pos_after, pos)
        self.velocity_cur = jnp.where(collides, vel_after, vel)


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
        default=Path("cache/bounce2d"),
        help="Directory to store logs in.",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed (for repeatability)"
    )
    parser.add_argument(
        "--starttime", type=float, default=0.0, help="Simulation start time (sec)"
    )
    parser.add_argument(
        "--simsteps",
        type=int,
        default=100,
        help="Number of timesteps to run simulation for",
    )
    parser.add_argument(
        "--dtime", type=float, default=1 / 30, help="Simulation timestep size (sec)"
    )
    parser.add_argument(
        "--compare-every",
        type=int,
        default=4,
        help="Apply loss every `--compare-every` frames.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of epochs to run parameter optimization for",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=Path("sampledata/sphere.obj"),
        help="Path to template sphere mesh (.obj) file.",
    )
    parser.add_argument("--log", action="store_true", help="Save log files.")

    args = parser.parse_args()

    if args.compare_every >= args.simsteps:
        raise ValueError(
            f"Arg --compare-every cannot be greater than or equal to {args.simsteps}."
        )

    renderer = SoftRenderer(camera_mode="look_at")
    camera_distance = 15.0
    elevation = 0.0
    azimuth = 0.0
    renderer.set_eye_from_angles(camera_distance, elevation, azimuth)

    # GT parameters
    position_initial_gt = jnp.array([-7.5, 0.0])
    radius_gt = 0.75
    speed_gt = 1.0
    height_gt = 5.0
    gravity_gt = -9.0

    ball2d_gt = BouncingBall2D(
        pos=position_initial_gt,
        radius=radius_gt,
        height=height_gt,
        speed=speed_gt,
        gravity=gravity_gt,
    )

    sphere = TriangleMesh.from_obj(args.template)
    vertices_gt = meshutils.normalize_vertices(
        sphere.vertices[None, :], scale_factor=radius_gt
    )
    faces = sphere.faces[None, :]
    textures_red = jnp.concatenate(
        (
            jnp.ones((1, faces.shape[1], 2, 1), dtype=jnp.float32),
            jnp.zeros((1, faces.shape[1], 2, 1), dtype=jnp.float32),
            jnp.zeros((1, faces.shape[1], 2, 1), dtype=jnp.float32),
        ),
        axis=-1,
    )

    logdir = Path("cache/bounce2d") / args.expid
    if args.log:
        logdir.mkdir(exist_ok=True)

    traj_gt = []
    imgs_gt = []
    print("Rendering GT images...")
    for t in trange(args.simsteps):
        ball2d_gt.step(args.dtime)
        traj_gt.append(ball2d_gt.position_cur)
        pos = jnp.zeros(3, dtype=vertices_gt.dtype)
        pos = pos.at[0].set(ball2d_gt.position_cur[0])
        pos = pos.at[1].set(ball2d_gt.position_cur[1])
        _vertices = vertices_gt + pos
        imgs_gt.append(renderer.forward(_vertices, faces, textures_red))

    if args.log:
        write_imglist_to_gif(
            imgs_gt, logdir / "gt.gif", imgformat="rgba", verbose=False,
        )

    # Parameters to estimate
    speed_init = jnp.array([3.0], dtype=jnp.float32)
    gravity_init = jnp.array([-10.0], dtype=jnp.float32)

    params = {
        "speed_update": jnp.zeros(1, dtype=jnp.float32),
        "gravity_update": jnp.zeros(1, dtype=jnp.float32),
    }
    opt_state = _adam_init(params)

    def loss_fn(params):
        speed_cur = speed_init + params["speed_update"]
        gravity_cur = gravity_init + params["gravity_update"]
        ball2d = BouncingBall2D(
            pos=position_initial_gt,
            radius=radius_gt,
            height=height_gt,
            speed=speed_cur[0],
            gravity=gravity_cur[0],
        )
        imgs_est = []
        for t in range(args.simsteps):
            ball2d.step(args.dtime)
            pos = jnp.zeros(3, dtype=vertices_gt.dtype)
            pos = pos.at[0].set(ball2d.position_cur[0])
            pos = pos.at[1].set(ball2d.position_cur[1])
            _vertices = vertices_gt + pos
            imgs_est.append(renderer.forward(_vertices, faces, textures_red))
        return sum(
            jnp.mean((est - gt) ** 2)
            for est, gt in zip(
                imgs_est[:: args.compare_every], imgs_gt[:: args.compare_every]
            )
        ) / len(imgs_est[:: args.compare_every])

    grad_fn = jax.value_and_grad(loss_fn)

    for e in trange(args.epochs):
        loss_val, grads = grad_fn(params)
        params, opt_state = _adam_step(params, grads, opt_state, lr=1e-1)

        speed_cur = speed_init + params["speed_update"]
        gravity_cur = gravity_init + params["gravity_update"]
        tqdm.write(
            f"Loss: {float(loss_val):.5f} "
            f"Speed error: {float(speed_cur[0] - speed_gt):.5f} "
            f"Gravity error: {float(gravity_cur[0] - gravity_gt):.5f}"
        )

        if args.log and e == 0:
            ball2d = BouncingBall2D(
                pos=position_initial_gt,
                radius=radius_gt,
                height=height_gt,
                speed=float(speed_cur[0]),
                gravity=float(gravity_cur[0]),
            )
            imgs_est = []
            for t in range(args.simsteps):
                ball2d.step(args.dtime)
                pos = jnp.zeros(3, dtype=vertices_gt.dtype)
                pos = pos.at[0].set(ball2d.position_cur[0])
                pos = pos.at[1].set(ball2d.position_cur[1])
                _vertices = vertices_gt + pos
                imgs_est.append(renderer.forward(_vertices, faces, textures_red))
            write_imglist_to_gif(
                imgs_est, logdir / "init.gif", imgformat="rgba", verbose=False,
            )

    if args.log:
        ball2d = BouncingBall2D(
            pos=position_initial_gt,
            radius=radius_gt,
            height=height_gt,
            speed=float(speed_cur[0]),
            gravity=float(gravity_cur[0]),
        )
        imgs_opt = []
        for t in range(args.simsteps):
            ball2d.step(args.dtime)
            pos = jnp.zeros(3, dtype=vertices_gt.dtype)
            pos = pos.at[0].set(ball2d.position_cur[0])
            pos = pos.at[1].set(ball2d.position_cur[1])
            _vertices = vertices_gt + pos
            imgs_opt.append(renderer.forward(_vertices, faces, textures_red))
        write_imglist_to_gif(
            imgs_opt, logdir / "opt.gif", imgformat="rgba", verbose=False,
        )
