"""
DIB-R: Sanity check
Example script that uses DIB-R to optimize the texture for a given mesh.
"""

import argparse
from pathlib import Path

import imageio
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm, trange

from gradsim.renderutils import TriangleMesh
from gradsim.renderutils.dibr.renderer import Renderer as DIBRenderer


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
        default=20,
        help="Number of iterations to run optimization for.",
    )
    parser.add_argument(
        "--no-viz", action="store_true", help="Skip visualization steps."
    )
    args = parser.parse_args()

    renderer = DIBRenderer(256, 256, mode="VertexColor")

    camera_distance = 2.0
    elevation = 30.0
    azimuth = 0.0

    DATA_DIR = Path(__file__).parent / "sampledata"

    logdir = Path(__file__).parent / "cache" / "dibr"
    logdir.mkdir(exist_ok=True)

    mesh = TriangleMesh.from_obj(DATA_DIR / "banana.obj")

    progressfile = logdir / "texture_optimization_progress.gif"
    outfile = logdir / "texture_optimization_output.gif"

    vertices = mesh.vertices[None, :, :]
    faces = mesh.faces[None, :, :]
    textures_init = jnp.ones((1, vertices.shape[-2], 3), dtype=jnp.float32)

    vertices_max = vertices.max()
    vertices_min = vertices.min()
    vertices_middle = (vertices_max + vertices_min) / 2.0
    vertices = vertices - vertices_middle
    coef = 5
    vertices = vertices * coef

    img_target = jnp.array(
        imageio.imread(DATA_DIR / "banana.png").astype(np.float32) / 255
    )
    img_target = img_target[None, ...]  # (1, H, W, 4)

    params = {"textures": textures_init}
    opt_state = _adam_init(params)

    renderer.set_look_at_parameters([90 - azimuth], [elevation], [camera_distance])

    def loss_fn(params):
        textures = jax.nn.sigmoid(params["textures"])
        img_pred, alpha, _ = renderer.forward(
            points=[vertices, faces[0]], colors_bxpx3=textures
        )
        return jnp.mean((img_pred[..., :3] - img_target[..., :3]) ** 2)

    grad_fn = jax.value_and_grad(loss_fn)

    if not args.no_viz:
        writer = imageio.get_writer(progressfile, mode="I")
    for i in trange(args.iters):
        loss_val, grads = grad_fn(params)
        params, opt_state = _adam_step(params, grads, opt_state, lr=1.0)
        if i % 5 == 0:
            tqdm.write(f"Loss: {float(loss_val):.5}")
            if not args.no_viz:
                textures = jax.nn.sigmoid(params["textures"])
                img_pred, _, _ = renderer.forward(
                    points=[vertices, faces[0]], colors_bxpx3=textures
                )
                img = np.array(img_pred[0])
                writer.append_data((255 * img).astype(np.uint8))
    if not args.no_viz:
        writer.close()

        writer = imageio.get_writer(outfile, mode="I")
        for az in trange(0, 360, 6):
            renderer.set_look_at_parameters(
                [90 - az], [elevation], [camera_distance]
            )
            textures = jax.nn.sigmoid(params["textures"])
            img_pred, _, _ = renderer.forward(
                points=[vertices, faces[0]], colors_bxpx3=textures
            )
            img = np.array(img_pred[0])
            writer.append_data((255 * img).astype(np.uint8))
        writer.close()
