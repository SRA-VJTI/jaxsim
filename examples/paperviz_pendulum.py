"""
Optimize parameters of a simple pendulum from video.
"""

import argparse
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from tqdm import tqdm, trange

from gradsim.bodies import SimplePendulum
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
    parser.add_argument("--logdir", type=str, default=Path("cache/simple_pendulum"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--starttime", type=float, default=0.0)
    parser.add_argument("--simsteps", type=int, default=50)
    parser.add_argument("--dtime", type=float, default=0.1)
    parser.add_argument("--gravity", type=float, default=9.91)
    parser.add_argument("--length", type=float, default=1.0)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--compare-every", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--template", type=str, default=Path("sampledata/sphere.obj"))
    parser.add_argument("--optimize-length", action="store_true")
    parser.add_argument("--optimize-gravity", action="store_true")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    if args.compare_every >= args.simsteps:
        raise ValueError(f"Arg --compare-every cannot be >= {args.simsteps}.")

    sphere = TriangleMesh.from_obj(args.template)
    vertices_gt = meshutils.normalize_vertices(
        sphere.vertices[None, :], scale_factor=args.radius
    )
    faces_gt = sphere.faces[None, :]
    textures = jnp.concatenate(
        [
            jnp.ones((1, faces_gt.shape[1], 2, 1), dtype=jnp.float32),
            jnp.zeros((1, faces_gt.shape[1], 2, 1), dtype=jnp.float32),
            jnp.zeros((1, faces_gt.shape[1], 2, 1), dtype=jnp.float32),
        ],
        axis=-1,
    )

    renderer = SoftRenderer(camera_mode="look_at")
    renderer.set_eye_from_angles(8.0, 0.0, 0.0)

    pendulum_gt = SimplePendulum(
        args.mass, args.radius, args.gravity, args.length, args.damping
    )
    theta_init = jnp.array([0.0, 3.0])
    times = jnp.arange(
        args.starttime,
        args.starttime + args.simsteps * args.dtime,
        args.dtime,
    )
    numsteps = int(times.size)
    ret = odeint(lambda y, t: pendulum_gt.forward(t, y), theta_init, times)

    theta1_gt = ret[:, 0]
    x_gt = args.length * jnp.sin(theta1_gt)
    y_gt = -args.length * jnp.cos(theta1_gt)
    pos_gt = jnp.stack([x_gt, y_gt, jnp.zeros_like(x_gt)], axis=-1)

    imgs_gt = []
    print("Rendering GT images...")
    for i in trange(numsteps):
        rgba = renderer.forward(vertices_gt + pos_gt[i], faces_gt, textures)
        imgs_gt.append(rgba)

    logdir = Path(args.logdir) / args.expid
    if args.log:
        logdir.mkdir(exist_ok=True, parents=True)
        write_imglist_to_gif(imgs_gt, logdir / "gt.gif", imgformat="rgba", verbose=False)

    params = {}
    if args.optimize_length:
        params["length"] = jnp.array([0.5])
    if args.optimize_gravity:
        params["gravity"] = jnp.array([5.0])

    opt_state = _adam_init(params)

    def loss_fn(params):
        length_cur = params["length"][0] if "length" in params else jnp.array(args.length)
        gravity_cur = params["gravity"][0] if "gravity" in params else jnp.array(args.gravity)
        pend = SimplePendulum(args.mass, args.radius, gravity_cur, length_cur, args.damping)
        ret = odeint(lambda y, t: pend.forward(t, y), theta_init, times)
        theta1 = ret[:, 0]
        x = length_cur * jnp.sin(theta1)
        y = -length_cur * jnp.cos(theta1)
        pos = jnp.stack([x, y, jnp.zeros_like(x)], axis=-1)
        imgs_est = [renderer.forward(vertices_gt + pos[i], faces_gt, textures)
                    for i in range(numsteps)]
        pairs = list(zip(imgs_est[:: args.compare_every], imgs_gt[:: args.compare_every]))
        return sum(jnp.mean((e - g) ** 2) for e, g in pairs) / len(pairs)

    grad_fn = jax.value_and_grad(loss_fn)
    losses = []
    best_loss_so_far = 1e6

    for e in trange(args.epochs):
        loss_val, grads = grad_fn(params)
        params, opt_state = _adam_step(params, grads, opt_state, lr=1e-1)
        losses.append(float(loss_val))

        if float(loss_val) <= best_loss_so_far:
            best_loss_so_far = float(loss_val)
            if args.log:
                lc = float(params.get("length", jnp.array([args.length]))[0])
                gc = float(params.get("gravity", jnp.array([args.gravity]))[0])
                pend = SimplePendulum(args.mass, args.radius, gc, lc, args.damping)
                ret2 = odeint(lambda y, t: pend.forward(t, y), theta_init, times)
                theta1 = ret2[:, 0]
                x = lc * jnp.sin(theta1)
                y = -lc * jnp.cos(theta1)
                pos = jnp.stack([x, y, jnp.zeros_like(x)], axis=-1)
                imgs_best = [renderer.forward(vertices_gt + pos[i], faces_gt, textures)
                             for i in range(numsteps)]
                write_imglist_to_gif(imgs_best, logdir / "best.gif", imgformat="rgba", verbose=False)

        if args.log and e == 0:
            write_imglist_to_gif(imgs_gt, logdir / "init.gif", imgformat="rgba", verbose=False)
        if args.log and e == args.epochs - 1:
            write_imglist_to_gif(imgs_gt, logdir / "opt.gif", imgformat="rgba", verbose=False)

        length_err = abs(float(params.get("length", jnp.array([args.length]))[0]) - args.length)
        tqdm.write(f"Loss: {float(loss_val):.5f}, length_error: {length_err:.5f}")

    if args.log:
        np.savetxt(logdir / "losses.txt", losses)
