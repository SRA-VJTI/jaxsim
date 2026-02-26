# Example to sanity check whether SoftRas works.

import argparse
from pathlib import Path

import imageio
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import trange

from gradsim.renderutils import SoftRenderer, TriangleMesh

# Example script that uses SoftRas to render an image, given a mesh input

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stride",
        type=int,
        default=6,
        help="Rotation (in degrees) between successive render azimuth angles.",
    )
    parser.add_argument(
        "--no-viz", action="store_true", help="Skip visualization steps."
    )
    args = parser.parse_args()

    # Initialize the soft rasterizer.
    renderer = SoftRenderer(camera_mode="look_at")

    # Camera settings.
    camera_distance = 2.0
    elevation = 30.0

    # Directory in which sample data is located.
    DATA_DIR = Path(__file__).parent / "sampledata"

    # Read in the input mesh.
    mesh = TriangleMesh.from_obj(DATA_DIR / "banana.obj")

    # Output filename (to write out a rendered .gif to).
    outfile = "cache/softras_render.gif"
    Path("cache").mkdir(exist_ok=True)

    # Extract the vertices, faces, and texture the mesh (currently color with yellow).
    vertices = mesh.vertices[None, :, :]
    faces = mesh.faces[None, :, :]
    # Initialize all faces to yellow (to color the banana)!
    textures = jnp.concatenate(
        [
            jnp.ones((1, faces.shape[1], 2, 1), dtype=jnp.float32),
            jnp.ones((1, faces.shape[1], 2, 1), dtype=jnp.float32),
            jnp.zeros((1, faces.shape[1], 2, 1), dtype=jnp.float32),
        ],
        axis=-1,
    )

    # Translate the mesh such that it is centered at the origin.
    vertices_max = vertices.max()
    vertices_min = vertices.min()
    vertices_middle = (vertices_max + vertices_min) / 2.0
    vertices = vertices - vertices_middle
    # Scale the vertices slightly (so that they occupy a sizeable image area).
    coef = 5
    vertices = vertices * coef

    # Loop over a set of azimuth angles, and render the image.
    print("Rendering using softras...")
    if not args.no_viz:
        writer = imageio.get_writer(outfile, mode="I")
    for azimuth in trange(0, 360, args.stride):
        renderer.set_eye_from_angles(camera_distance, elevation, azimuth)
        rgba = renderer.forward(vertices, faces, textures)
        if not args.no_viz:
            img = np.array(rgba[0]).transpose(1, 2, 0)
            writer.append_data((255 * img).astype(np.uint8))
    if not args.no_viz:
        writer.close()
