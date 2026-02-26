from pathlib import Path

import imageio
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import jax
import jax.numpy as jnp
from tqdm import tqdm, trange


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


def rollrow():
    return lambda x, shift: jnp.roll(x, shift, axis=0)


def rollcol():
    return lambda x, shift: jnp.roll(x, shift, axis=1)


def project(vx, vy, params):
    """Project the velocity field to be approximately mass-conserving, using a few
    iterations of Gauss-Siedel. """
    p = jnp.zeros(vx.shape, dtype=vx.dtype)
    h = 1.0 / vx.shape[0]
    div = (
        -0.5
        * h
        * (rollrow()(vx, -1) - rollrow()(vx, 1) + rollcol()(vy, -1) - rollcol()(vy, 1))
    )

    for k in range(6):
        p = (
            div
            + rollrow()(p, 1)
            + rollrow()(p, -1)
            + rollcol()(p, 1)
            + rollcol()(p, -1)
        ) / 4.0

    vx = vx - 0.5 * (rollrow()(p, -1) - rollrow()(p, 1)) / h
    vy = vy - 0.5 * (rollcol()(p, -1) - rollcol()(p, 1)) / h
    return vx, vy


def advect(f, vx, vy, params):
    """Move field f according to velocities vx, vy using an implicit Euler
    integrator. """
    rows, cols = f.shape
    cell_ys_orig, cell_xs_orig = jnp.meshgrid(jnp.arange(rows), jnp.arange(cols), indexing='ij')
    cell_xs = jnp.transpose(cell_xs_orig).astype(jnp.float32)
    cell_ys = jnp.transpose(cell_ys_orig).astype(jnp.float32)
    center_xs = (cell_xs - vx).flatten()
    center_ys = (cell_ys - vy).flatten()

    # Compute indices of source cells
    left_ix = jnp.floor(center_xs).astype(jnp.int32)
    top_ix = jnp.floor(center_ys).astype(jnp.int32)
    rw = center_xs - left_ix.astype(jnp.float32)  # Relative weight of right-hand cells
    bw = center_ys - top_ix.astype(jnp.float32)   # Relative weight of bottom cells
    left_ix = left_ix % rows   # Wrap around edges of simulation
    right_ix = (left_ix + 1) % rows
    top_ix = top_ix % cols
    bot_ix = (top_ix + 1) % cols

    # A linearly-weighted sum of the 4 surrounding cells
    flat_f = (1 - rw) * (
        (1 - bw) * f[left_ix, top_ix] + bw * f[left_ix, bot_ix]
    ) + rw * ((1 - bw) * f[right_ix, top_ix] + bw * f[right_ix, bot_ix])

    return flat_f.reshape((rows, cols))


def forward(iteration, smoke, vx, vy, output, params):
    for t in range(1, params["steps"]):
        vx_updated = advect(vx, vx, vy, params)
        vy_updated = advect(vy, vx, vy, params)
        vx, vy = project(vx_updated, vy_updated, params)
        smoke = advect(smoke, vx, vy, params)
        if output:
            smoke2d_path = Path("cache/smoke2d")
            smoke2d_path.mkdir(parents=True, exist_ok=True)
            if plt is not None:
                plt.imsave(
                    smoke2d_path / f"{t:03d}.png", 255 * jnp.array(smoke)
                )
    return smoke


if __name__ == "__main__":

    numiters = 50
    gridsize = 110
    dx = 1.0 / gridsize
    lr = 1e-1
    printevery = 1

    params = {}
    params["steps"] = 100

    target_smoke_img = imageio.imread(Path("sampledata/smoke_target.png")) / 255.0

    target = jnp.array(target_smoke_img, dtype=jnp.float32)

    initial_smoke = jnp.zeros_like(target)
    initial_smoke = initial_smoke.at[2 * gridsize // 3:, :].set(1.0)

    opt_params = {
        "vx": jnp.zeros((gridsize, gridsize), dtype=jnp.float32),
        "vy": jnp.zeros((gridsize, gridsize), dtype=jnp.float32),
    }
    opt_state = _adam_init(opt_params)

    def loss_fn(p):
        smoke = forward(0, initial_smoke, p["vx"], p["vy"], False, params)
        return jnp.mean((smoke - target) ** 2)

    grad_fn = jax.value_and_grad(loss_fn)

    for iteration in trange(numiters):
        loss_val, grads = grad_fn(opt_params)
        opt_params, opt_state = _adam_step(opt_params, grads, opt_state, lr=lr)

        if iteration % printevery == 0:
            tqdm.write(f"Iter {iteration} Loss: {float(loss_val):.8}")

    # Final render
    forward(
        numiters - 1,
        initial_smoke,
        opt_params["vx"],
        opt_params["vy"],
        True,
        params,
    )
