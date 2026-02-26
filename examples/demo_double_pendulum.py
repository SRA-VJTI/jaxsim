"""
Optimize parameters of a double-pendulum from video.
"""

import argparse
import math
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from tqdm import tqdm, trange

from gradsim.bodies import DoublePendulum
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--expid", type=str, default="default")
    parser.add_argument("--logdir", type=str, default=Path("cache/double_pendulum"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--starttime", type=float, default=0.0)
    parser.add_argument("--simsteps", type=int, default=50)
    parser.add_argument("--dtime", type=float, default=1 / 30)
    parser.add_argument("--gravity", type=float, default=10.0)
    parser.add_argument("--length1", type=float, default=1.0)
    parser.add_argument("--length2", type=float, default=1.0)
    parser.add_argument("--mass1", type=float, default=1.0)
    parser.add_argument("--mass2", type=float, default=1.0)
    parser.add_argument("--radius", type=float, default=0.25)
    parser.add_argument("--compare-every", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--template", type=str, default=Path("sampledata/sphere.obj"))
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    if args.compare_every >= args.simsteps:
        raise ValueError(f"Arg --compare-every cannot be >= {args.simsteps}.")

    renderer = SoftRenderer(camera_mode="look_at")
    renderer.set_eye_from_angles(8.0, 0.0, 0.0)

    sphere = TriangleMesh.from_obj(args.template)
    vertices_gt = meshutils.normalize_vertices(
        sphere.vertices[None, :], scale_factor=args.radius
    )
    faces = sphere.faces[None, :]
    textures_red = jnp.concatenate(
        [
            jnp.ones((1, faces.shape[1], 2, 1), dtype=jnp.float32),
            jnp.zeros((1, faces.shape[1], 2, 1), dtype=jnp.float32),
            jnp.zeros((1, faces.shape[1], 2, 1), dtype=jnp.float32),
        ],
        axis=-1,
    )
    textures_blue = jnp.concatenate(
        [
            jnp.zeros((1, faces.shape[1], 2, 1), dtype=jnp.float32),
            jnp.zeros((1, faces.shape[1], 2, 1), dtype=jnp.float32),
            jnp.ones((1, faces.shape[1], 2, 1), dtype=jnp.float32),
        ],
        axis=-1,
    )

    logdir = Path(args.logdir) / args.expid
    if args.log:
        logdir.mkdir(exist_ok=True, parents=True)

    double_pendulum_gt = DoublePendulum(
        args.length1, args.length2, args.mass1, args.mass2, args.gravity
    )
    times = jnp.arange(
        args.starttime,
        args.starttime + args.simsteps * args.dtime,
        args.dtime,
    )
    numsteps = int(times.size)
    y0 = jnp.array([3 * math.pi / 7, 0.0, 3 * math.pi / 4, 0.0])

    y_gt = odeint(lambda y, t: double_pendulum_gt.forward(t, y), y0, times)

    edrift = 0.1
    einit = double_pendulum_gt.compute_energy(y0)
    if float(jnp.max(jnp.sum(jnp.abs(double_pendulum_gt.compute_energy(y_gt) - einit)))) > edrift:
        print(f"[WARNING] Maximum energy drift of {edrift} exceeded!")

    theta1_gt, theta2_gt = y_gt[:, 0], y_gt[:, 2]
    x1 = double_pendulum_gt.length1 * jnp.sin(theta1_gt)
    y1 = -double_pendulum_gt.length1 * jnp.cos(theta1_gt)
    x2 = x1 + double_pendulum_gt.length2 * jnp.sin(theta2_gt)
    y2 = y1 - double_pendulum_gt.length2 * jnp.cos(theta2_gt)

    pos1_gt = jnp.stack([x1, y1, jnp.zeros_like(x1)], axis=-1)
    pos2_gt = jnp.stack([x2, y2, jnp.zeros_like(x2)], axis=-1)

    imgs1_gt, imgs2_gt = [], []
    print("Rendering GT images...")
    for i in trange(numsteps):
        imgs1_gt.append(renderer.forward(vertices_gt + pos1_gt[i], faces, textures_red))
        imgs2_gt.append(renderer.forward(vertices_gt + pos2_gt[i], faces, textures_blue))

    if args.log:
        imgs_gt = [0.5 * (b1 + b2) for b1, b2 in zip(imgs1_gt, imgs2_gt)]
        write_imglist_to_gif(imgs_gt, logdir / "gt.gif", imgformat="rgba", verbose=False)

    params = {"l1": jnp.array([1.0]), "l2": jnp.array([1.7])}
    opt_state = _adam_init(params)

    def loss_fn(params):
        l1, l2 = params["l1"][0], params["l2"][0]
        dp = DoublePendulum(l1, l2, args.mass1, args.mass2, args.gravity)
        y = odeint(lambda y, t: dp.forward(t, y), y0, times)
        th1, th2 = y[:, 0], y[:, 2]
        _x1 = l1 * jnp.sin(th1)
        _y1 = -l1 * jnp.cos(th1)
        _x2 = _x1 + l2 * jnp.sin(th2)
        _y2 = _y1 - l2 * jnp.cos(th2)
        p1 = jnp.stack([_x1, _y1, jnp.zeros_like(_x1)], axis=-1)
        p2 = jnp.stack([_x2, _y2, jnp.zeros_like(_x2)], axis=-1)
        imgs1 = [renderer.forward(vertices_gt + p1[i], faces, textures_red) for i in range(numsteps)]
        imgs2 = [renderer.forward(vertices_gt + p2[i], faces, textures_blue) for i in range(numsteps)]
        pairs = (list(zip(imgs1[:: args.compare_every], imgs1_gt[:: args.compare_every])) +
                 list(zip(imgs2[:: args.compare_every], imgs2_gt[:: args.compare_every])))
        return sum(jnp.mean((e - g) ** 2) for e, g in pairs) / len(pairs)

    grad_fn = jax.value_and_grad(loss_fn)
    best_loss = 1e6

    for e in trange(args.epochs):
        loss_val, grads = grad_fn(params)
        params, opt_state = _adam_step(params, grads, opt_state, lr=1e-2)
        l1_err = abs(float(params["l1"][0]) - args.length1)
        l2_err = abs(float(params["l2"][0]) - args.length2)
        tqdm.write(
            f"Loss: {float(loss_val):.5f}, l1_error: {l1_err:.5f}, l2_error: {l2_err:.5f}"
        )
        if float(loss_val) <= best_loss:
            best_loss = float(loss_val)
            if args.log:
                # recompute estimated images at best params
                l1, l2 = float(params["l1"][0]), float(params["l2"][0])
                dp = DoublePendulum(l1, l2, args.mass1, args.mass2, args.gravity)
                y = odeint(lambda y, t: dp.forward(t, y), y0, times)
                th1, th2 = y[:, 0], y[:, 2]
                _x1 = l1 * jnp.sin(th1); _y1 = -l1 * jnp.cos(th1)
                _x2 = _x1 + l2 * jnp.sin(th2); _y2 = _y1 - l2 * jnp.cos(th2)
                p1 = jnp.stack([_x1, _y1, jnp.zeros_like(_x1)], axis=-1)
                p2 = jnp.stack([_x2, _y2, jnp.zeros_like(_x2)], axis=-1)
                imgs_best = [0.5 * (renderer.forward(vertices_gt + p1[i], faces, textures_red) +
                                    renderer.forward(vertices_gt + p2[i], faces, textures_blue))
                             for i in range(numsteps)]
                write_imglist_to_gif(imgs_best, logdir / "best.gif", imgformat="rgba", verbose=False)
