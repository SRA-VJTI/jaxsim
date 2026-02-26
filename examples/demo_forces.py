from pathlib import Path

import imageio
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

    # Output (gif) file path
    outfile = Path("cache/demoforces.gif")
    outfile.parent.mkdir(exist_ok=True)

    # Load a body (from a triangle mesh obj file).
    mesh = TriangleMesh.from_obj(Path("sampledata/cube.obj"))
    vertices = meshutils.normalize_vertices(mesh.vertices[None, :])
    faces = mesh.faces[None, :]
    textures = jnp.concatenate(
        [
            jnp.ones((1, faces.shape[1], 2, 1), dtype=jnp.float32),
            jnp.ones((1, faces.shape[1], 2, 1), dtype=jnp.float32),
            jnp.zeros((1, faces.shape[1], 2, 1), dtype=jnp.float32),
        ],
        axis=-1,
    )
    masses = 0.1 * jnp.ones(vertices.shape[1], dtype=jnp.float32)
    body = RigidBody(vertices[0], masses=masses)

    # Create a force that applies gravity (g = 10 metres / second^2).
    gravity = ConstantForce(direction=jnp.array([0.0, -1.0, 0.0]))

    # Add this force to the body.
    body.add_external_force(gravity, application_points=[0, 1])

    # Initialize the simulator with the body at the origin.
    sim = Simulator([body])

    # Initialize the renderer.
    renderer = SoftRenderer(camera_mode="look_at")
    camera_distance = 8.0
    elevation = 30.0
    azimuth = 0.0
    renderer.set_eye_from_angles(camera_distance, elevation, azimuth)

    # Run the simulation.
    writer = imageio.get_writer(outfile, mode="I")
    for i in trange(20):
        sim.step()
        rgba = renderer.forward(body.get_world_vertices()[None, :], faces, textures)
        img = np.array(rgba[0]).transpose(1, 2, 0)
        writer.append_data((255 * img).astype(np.uint8))
    writer.close()
