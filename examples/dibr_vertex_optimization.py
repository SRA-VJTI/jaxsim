"""
DIB-R: Sanity check
Uses DIB-R to optimize the vertices a given mesh to match a rendered image.
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
from gradsim.renderutils.dibr.utils.sphericalcoord import get_spherical_coords_x

# Example script that uses DIB-R to deform a sphere mesh to approximate
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


def _build_laplacian_matrix(nv, faces_np):
    """Build the Laplacian matrix using dense numpy (replaces torch.sparse.FloatTensor)."""
    L = np.zeros((nv, nv), dtype=np.float32)
    # Diagonal entries: 2 identity entries * 0.5 = 1.0 per vertex
    np.fill_diagonal(L, 1.0)
    v1, v2, v3 = faces_np[:, 0], faces_np[:, 1], faces_np[:, 2]
    np.add.at(L, (v1, v2), 0.5)
    np.add.at(L, (v1, v3), 0.5)
    np.add.at(L, (v2, v1), 0.5)
    np.add.at(L, (v2, v3), 0.5)
    np.add.at(L, (v3, v2), 0.5)
    np.add.at(L, (v3, v1), 0.5)
    return jnp.array(L)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iters",
        type=int,
        default=200,
        help="Number of iterations to run optimization for.",
    )
    parser.add_argument(
        "--no-viz", action="store_true", help="Skip visualization steps."
    )
    parser.add_argument(
        "--renderer",
        type=str,
        choices=["vc", "lam", "sh"],
        default="vc",
        help='Type of color handling to used in renderer. "vc" uses VertexColor mode. '
        '"lam" uses Lambertian mode. "sh" uses SphericalHarmonics.',
    )
    args = parser.parse_args()

    if args.renderer == "vc":
        renderer = DIBRenderer(256, 256, mode="VertexColor")
    elif args.renderer == "lam":
        renderer = DIBRenderer(256, 256, mode="Lambertian")
    elif args.renderer == "sh":
        renderer = DIBRenderer(256, 256, mode="SphericalHarmonics")
    else:
        raise ValueError(
            'Renderer mode must be one of ["VertexColor", "Lambertian"'
            f' or "SphericalHarmonics"]. Got {args.renderer} instead.'
        )

    camera_distance = 2.0
    elevation = 30.0
    azimuth = 0.0

    DATA_DIR = Path(__file__).parent / "sampledata"

    logdir = Path(__file__).parent / "cache" / "dibr"
    logdir.mkdir(exist_ok=True)

    mesh = TriangleMesh.from_obj(DATA_DIR / "dibr_sphere.obj")

    progressfile = logdir / "vertex_optimization_progress.gif"
    outfile = logdir / "vertex_optimization_output.gif"

    vertices = mesh.vertices[None, :, :]
    faces = mesh.faces[None, :, :]

    textures = jnp.stack(
        (
            jnp.ones((1, vertices.shape[-2]), dtype=jnp.float32),
            jnp.ones((1, vertices.shape[-2]), dtype=jnp.float32),
            jnp.zeros((1, vertices.shape[-2]), dtype=jnp.float32),
        ),
        axis=-1,
    )

    uv, texture_img, lightparam = None, None, None
    if args.renderer in ["lam", "sh"]:
        uv_np = get_spherical_coords_x(np.array(vertices[0]))
        uv = jnp.array(uv_np)[None, :, :] / 255.0
        texture_img = jnp.zeros((1, 3, 128, 128), dtype=jnp.float32)
        texture_img = texture_img.at[:, 0, :, :].set(1.0)
        texture_img = texture_img.at[:, 1, :, :].set(1.0)
        if args.renderer == "sh":
            lightparam = jnp.zeros(9, dtype=jnp.float32)

    img_target = jnp.array(
        imageio.imread(DATA_DIR / "banana.png").astype(np.float32) / 255
    )
    img_target = img_target[None, ...]

    # Pre-compute Laplacian matrix (static, based on face topology only)
    faces_np = np.array(mesh.faces)
    nv = mesh.vertices.shape[0]
    laplacian_matrix = _build_laplacian_matrix(nv, faces_np)

    params = {"update": jnp.zeros(vertices.shape, dtype=jnp.float32)}
    opt_state = _adam_init(params)

    renderer.set_look_at_parameters([90 - azimuth], [elevation], [camera_distance])

    def loss_fn(params):
        new_vertices = vertices + params["update"]
        if args.renderer == "vc":
            img_pred, alpha, _ = renderer.forward(
                points=[new_vertices, faces[0]], colors_bxpx3=textures
            )
        elif args.renderer == "lam":
            img_pred, alpha, _ = renderer.forward(
                points=[new_vertices, faces[0]],
                uv_bxpx2=uv,
                texture_bx3xthxtw=texture_img,
            )
        elif args.renderer == "sh":
            img_pred, alpha, _ = renderer.forward(
                points=[new_vertices, faces[0]],
                uv_bxpx2=uv,
                texture_bx3xthxtw=texture_img,
                lightparam=lightparam,
            )
        rgba = jnp.concatenate((img_pred, alpha), axis=-1)
        mse = jnp.mean((rgba - img_target) ** 2)
        # Laplacian regularization on vertex deformation
        deformation = new_vertices[0] - vertices[0]
        Ld = jnp.matmul(laplacian_matrix, deformation)
        laplacian_loss = jnp.mean(Ld ** 2)
        return mse + laplacian_loss

    grad_fn = jax.value_and_grad(loss_fn)

    if not args.no_viz:
        writer = imageio.get_writer(progressfile, mode="I")
    for i in trange(args.iters):
        loss_val, grads = grad_fn(params)
        params, opt_state = _adam_step(params, grads, opt_state, lr=0.01)
        if i % 20 == 0:
            tqdm.write(f"Loss: {float(loss_val):.5}")
            if not args.no_viz:
                new_vertices = vertices + params["update"]
                if args.renderer == "vc":
                    img_pred, alpha, _ = renderer.forward(
                        points=[new_vertices, faces[0]], colors_bxpx3=textures
                    )
                elif args.renderer == "lam":
                    img_pred, alpha, _ = renderer.forward(
                        points=[new_vertices, faces[0]],
                        uv_bxpx2=uv,
                        texture_bx3xthxtw=texture_img,
                    )
                elif args.renderer == "sh":
                    img_pred, alpha, _ = renderer.forward(
                        points=[new_vertices, faces[0]],
                        uv_bxpx2=uv,
                        texture_bx3xthxtw=texture_img,
                        lightparam=lightparam,
                    )
                img = np.array(img_pred[0])
                writer.append_data((255 * img).astype(np.uint8))
    if not args.no_viz:
        writer.close()

        writer = imageio.get_writer(outfile, mode="I")
        for azimuth in trange(0, 360, 6):
            renderer.set_look_at_parameters(
                [90 - azimuth], [elevation], [camera_distance]
            )
            new_vertices = vertices + params["update"]
            if args.renderer == "vc":
                img_pred, alpha, _ = renderer.forward(
                    points=[new_vertices, faces[0]], colors_bxpx3=textures
                )
            elif args.renderer == "lam":
                img_pred, alpha, _ = renderer.forward(
                    points=[new_vertices, faces[0]],
                    uv_bxpx2=uv,
                    texture_bx3xthxtw=texture_img,
                )
            elif args.renderer == "sh":
                img_pred, alpha, _ = renderer.forward(
                    points=[new_vertices, faces[0]],
                    uv_bxpx2=uv,
                    texture_bx3xthxtw=texture_img,
                    lightparam=lightparam,
                )
            img = np.array(img_pred[0])
            writer.append_data((255 * img).astype(np.uint8))
        writer.close()
