# Pure JAX TriangleMesh — OBJ loader returning jnp arrays.
# Replaces kaolin.rep.TriangleMesh dependency.

import numpy as np
import jax.numpy as jnp


class TriangleMesh:
    """Minimal triangle mesh container backed by JAX arrays."""

    def __init__(self, vertices, faces):
        # vertices: jnp (V, 3) float32
        # faces:    jnp (F, 3) int32
        self.vertices = vertices
        self.faces = faces

    @classmethod
    def from_obj(cls, path):
        """Load a Wavefront OBJ file and return a TriangleMesh."""
        verts, faces = [], []
        with open(str(path)) as f:
            for line in f:
                parts = line.strip().split()
                if not parts or parts[0].startswith('#'):
                    continue
                if parts[0] == 'v':
                    verts.append([float(x) for x in parts[1:4]])
                elif parts[0] == 'f':
                    # Each entry is v, v/vt, v/vt/vn, or v//vn — take vertex index only.
                    indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                    if len(indices) == 3:
                        faces.append(indices)
                    elif len(indices) == 4:
                        # Triangulate quad fan-style
                        faces.append([indices[0], indices[1], indices[2]])
                        faces.append([indices[0], indices[2], indices[3]])
                    elif len(indices) > 4:
                        for i in range(1, len(indices) - 1):
                            faces.append([indices[0], indices[i], indices[i + 1]])

        vertices = jnp.array(np.array(verts, dtype=np.float32))
        faces    = jnp.array(np.array(faces, dtype=np.int32))
        return cls(vertices, faces)
