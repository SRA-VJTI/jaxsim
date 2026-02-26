import jax.numpy as jnp

from gradsim.bodies import RigidBody
from gradsim.engines import (EulerIntegratorWithContacts,
                              SemiImplicitEulerWithContacts)
from gradsim.forces import ConstantForce
from gradsim.simulator import Simulator

if __name__ == "__main__":

    cube_verts = jnp.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ],
        dtype=jnp.float32,
    )
    position = jnp.array([2.0, 2.0, 2.0])
    orientation = jnp.array([1.0, 0.0, 0.0, 0.0])
    cube = RigidBody(cube_verts + 1, position=position, orientation=orientation)
    force_magnitude = 10.0
    force_direction = jnp.array([0.0, -1.0, 0.0])
    gravity = ConstantForce(magnitude=force_magnitude, direction=force_direction)
    cube.add_external_force(gravity)

    sim_substeps = 32
    dtime = (1 / 30) / sim_substeps
    sim = Simulator([cube], engine=SemiImplicitEulerWithContacts(), dtime=dtime)

    print(cube.position)
    print("vertices at start:", cube.get_world_vertices())

    for i in range(800):
        sim.step()
        if i % sim_substeps == 0:
            print(cube.position)
