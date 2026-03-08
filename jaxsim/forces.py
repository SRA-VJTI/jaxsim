from abc import abstractmethod

import jax.numpy as jnp


class ExternalForce(object):
    """A generic external force to be applied to rigid bodies.

    Takes in a direction vector along which the force is applied, and the magnitude
    of the applied force. The `force()` method computes the force vector at the
    specified timestep.
    """

    def __init__(
        self,
        direction,
        magnitude=10.0,
        starttime=0.0,
        endtime=1e5,
    ):
        r"""Initialize an external force object with the specified direction and
        magnitude.

        Args:
            direction (jnp.ndarray): Direction of the applied force
                (shape: :math:`(3)`).
            magnitude (float): Magnitude of the applied force (default: 10.0).
            starttime (float): Time (in seconds) at which the force is first applied
                (default: 0.0).
            endtime (float): Time (in seconds) at which the force stops being applied
                (default: 1e5).
        """
        self.direction = jnp.asarray(direction)
        self.magnitude = magnitude
        self.starttime = starttime
        self.endtime = endtime

    def apply(self, time):
        r"""Return the force vector at the time specified.

        Args:
            time (float): Time (in seconds).
        """
        if time < self.starttime or time > self.endtime:
            return self.direction * 0
        else:
            return self.force_function(time)

    @abstractmethod
    def force_function(self, time, *args, **kwargs):
        raise NotImplementedError


class ConstantForce(ExternalForce):
    """A constant force, with specified start and end times. """

    def __init__(
        self,
        direction,
        magnitude=10.0,
        starttime=0.0,
        endtime=1e5,
    ):
        super().__init__(direction, magnitude, starttime, endtime)

    def force_function(self, time):
        return self.direction * self.magnitude


class Gravity(ConstantForce):
    """A constant, downward force. """

    def __init__(
        self,
        direction=None,
        magnitude=10.0,
        starttime=0.0,
        endtime=1e5,
    ):
        if direction is None:
            direction = jnp.array([0.0, 0.0, -1.0])
        super().__init__(direction, magnitude, starttime, endtime)


class XForce(ConstantForce):
    """A constant force along the X-axis. """

    def __init__(
        self,
        direction=None,
        magnitude=10.0,
        starttime=0.0,
        endtime=1e5,
    ):
        if direction is None:
            direction = jnp.array([1.0, 0.0, 0.0])
        super().__init__(direction, magnitude, starttime, endtime)


class YForce(ConstantForce):
    """A constant force along the Y-axis. """

    def __init__(
        self,
        direction=None,
        magnitude=10.0,
        starttime=0.0,
        endtime=1e5,
    ):
        if direction is None:
            direction = jnp.array([0.0, 1.0, 0.0])
        super().__init__(direction, magnitude, starttime, endtime)
