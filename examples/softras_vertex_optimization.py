# Example to sanity check whether SoftRas works.

import argparse
from pathlib import Path

import imageio
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm, trange

from gradsim.renderutils import SoftRenderer, TriangleMesh

# Example script that uses SoftRas to deform a sphere mesh to approximate
# the image of a banana.


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
    parser.add_argument(
        "--iters",
        type=int,
        default=2000,
        help="Number of iterations to run optimization for.",
    )
    parser.add_argument(
        "--no-viz", action="store_true", help="Skip visualization steps."
    )
    args = parser.parse_args()

    renderer = SoftRenderer(camera_mode="look_at")

    camera_distance = 2.0
    elevation = 30.0
    azimuth = 0.0

    DATA_DIR = Path(__file__).parent / "sampledata"

    mesh = TriangleMesh.from_obj(DATA_DIR / "sphere.obj")

    progressfile = "vertex_optimization_progress.gif"
    outfile = "vertex_optimization_output.gif"

    vertices = mesh.vertices[None, :, :]
    faces = mesh.faces[None, :, :]
    textures = jnp.concatenate(
        (
            jnp.ones((1, faces.shape[1], 2, 1), dtype=jnp.float32),
            jnp.ones((1, faces.shape[1], 2, 1), dtype=jnp.float32),
            jnp.zeros((1, faces.shape[1], 2, 1), dtype=jnp.float32),
        ),
        axis=-1,
    )

    img_target = jnp.array(
        imageio.imread(DATA_DIR / "banana.png").astype(np.float32) / 255
    )
    img_target = img_target[None, ...].transpose(0, 3, 1, 2)

    # Vertex update parameter, initialized near zero
    params = {"update": jnp.zeros_like(vertices)}
    opt_state = _adam_init(params)

    def loss_fn(params):
        new_vertices = vertices + params["update"]
        rgba = renderer.forward(new_vertices, faces, textures)
        return jnp.mean((rgba - img_target) ** 2)

    grad_fn = jax.value_and_grad(loss_fn)

    renderer.set_eye_from_angles(camera_distance, elevation, azimuth)

    if not args.no_viz:
        writer = imageio.get_writer(progressfile, mode="I")
    for i in trange(args.iters):
        loss_val, grads = grad_fn(params)
        params, opt_state = _adam_step(params, grads, opt_state, lr=0.01)
        if i % 20 == 0:
            tqdm.write(f"Loss: {float(loss_val):.5}")
            if not args.no_viz:
                new_vertices = vertices + params["update"]
                rgba = renderer.forward(new_vertices, faces, textures)
                img = np.array(rgba[0]).transpose(1, 2, 0)
                writer.append_data((255 * img).astype(np.uint8))
    if not args.no_viz:
        writer.close()

        writer = imageio.get_writer(outfile, mode="I")
        for az in trange(0, 360, 6):
            renderer.set_eye_from_angles(camera_distance, elevation, az)
            new_vertices = vertices + params["update"]
            rgba = renderer.forward(new_vertices, faces, textures)
            img = np.array(rgba[0]).transpose(1, 2, 0)
            writer.append_data((255 * img).astype(np.uint8))
        writer.close()
