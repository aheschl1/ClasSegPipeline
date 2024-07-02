from classeg.extensions.unstable_diffusion.forward_diffusers.diffusers import Diffuser, LinearDiffuser


class DiffusionScheduler:
    def __init__(self, diffuser: Diffuser, *args, **kwargs):
        self.diffuser = diffuser
        self._step = 0
        self._state_dict = {
            "class_type": type(self).__name__,
        }
        self.diffuser.set_max_t(self.compute_max_at_step(self._step))

    def compute_max_at_step(self, step: int) -> int:
        """
        Compute the maximum t to sample at the current step.
        """
        raise NotImplementedError("Implement in subclass")

    def step(self):
        self._step += 1
        self._state_dict["step"] = self._step
        self.diffuser.set_max_t(self.compute_max_at_step(self._step))

    def get_state_dict(self):
        return self._state_dict

    def load_state(self, state_dict):
        if not state_dict["class_type"] == type(self).__name__:
            raise ValueError(f"State dict class type {state_dict['class_type']} does not match {type(self).__name__}")

        self._state_dict = state_dict
        self._step = state_dict["step"]
        self.diffuser.set_max_t(self.compute_max_at_step(self._step))


class StepScheduler(DiffusionScheduler):
    def __init__(self, diffuser: Diffuser, step_size: int = 5, epochs_per_step: int = 5, initial_max=1, *args, **kwargs):
        self.step_size = step_size
        self.epochs_per_step = epochs_per_step
        self.initial_max = initial_max
        super().__init__(diffuser, *args, **kwargs)
        self._state_dict['initial_max'] = initial_max
    
    def load_state(self, state_dict):
        self.initial_max = state_dict.get('initial_max', 1)
        return super().load_state(state_dict)

    def compute_max_at_step(self, step: int) -> int:
        return min(self.diffuser.timesteps, self.step_size * (step // self.epochs_per_step) + self.initial_max)


if __name__ == "__main__":
    diffuser = LinearDiffuser(20)
    scheduler = StepScheduler(diffuser)
    ts = []
    for i in range(20):
        scheduler.step()
        ts.append(scheduler.diffuser.max_t_to_sample)
    print(ts)
