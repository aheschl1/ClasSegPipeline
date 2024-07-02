from classeg.extensions.unstable_diffusion.forward_diffusers.diffusers import Diffuser


class DiffusionScheduler:
    def __init__(self, diffuser: Diffuser, *args, **kwargs):
        self.diffuser = diffuser
        self._step = 0
        self._state_dict = {}

    def compute_max_at_step(self, step: int) -> int:
        """
        Compute the maximum t to sample at the current step.
        """
        raise NotImplementedError("Implement in subclass")

    def step(self):
        self._step += 1
        self.diffuser.set_max_t(self.compute_max_at_step(self._step))

    @property
    def state_dict(self):
        return self._state_dict

