import torch


class Diffuser:
    def __init__(
        self, timesteps: int, min_beta: float = 0.0002, max_beta: float = 0.999
    ):
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.timesteps = timesteps
        self._betas, self._alphas, self._alpha_bars = None, None, None
        self._betas = self.prepare_betas()
        self._alphas = 1.0 - self._betas
        self._alpha_bars = torch.cumprod(self._alphas, 0)

    def __call__(self, im: torch.Tensor, seg: torch.Tensor, t=None, noise_im=None, noise_seg=None):
        """
        Given data, randomly samples timesteps and return noise, noisy_data, timesteps.
        Keeps devices uniform.


        """
        if t is None:
            t = torch.randint(0, self.timesteps, (im.shape[0],)).long()
        if noise_im is None:
            noise_im = torch.randn_like(im).to(im.device)
        if noise_seg is None:
            noise_seg = torch.randn_like(seg).to(im.device)

        a_bar = self._alpha_bars[t].to(im.device)

        noisy_im = (
            a_bar.sqrt().reshape(im.shape[0], 1, 1, 1) * im
            + (1 - a_bar).sqrt().reshape(im.shape[0], 1, 1, 1) * noise_im
        )
        noisy_seg = (
            a_bar.sqrt().reshape(seg.shape[0], 1, 1, 1) * seg
            + (1 - a_bar).sqrt().reshape(seg.shape[0], 1, 1, 1) * noise_seg
        )

        return noise_im, noise_seg, noisy_im, noisy_seg, t.to(im.device)

    def inference_call(
        self, 
        im: torch.Tensor, 
        seg:torch.Tensor,
        predicted_noise_im: torch.Tensor, 
        predicted_noise_seg: torch.Tensor, 
        t: int, clamp=False
    ):
        """
        For use in inference mode
        If clamp is true, clamps data between -1 and 1
        """
        alpha_t = self._alphas[t]
        alpha_t_bar = self._alpha_bars[t]
        data_im = (1 / alpha_t.sqrt()) * (
            im - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * predicted_noise_im
        )
        if clamp:
            data_im = torch.clip(data_im, -5, 5)
        if t > 0:
            z = torch.randn_like(predicted_noise_im).to(predicted_noise_im.device)
            beta_t = self._betas[t]
            sigma_t = beta_t.sqrt()
            data_im = data_im + sigma_t * z

        # SEG
        data_seg = (1 / alpha_t.sqrt()) * (
            seg - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * predicted_noise_seg
        )
        if clamp:
            data_seg = torch.clip(data_seg, -5, 5)
        if t > 0:
            z = torch.randn_like(predicted_noise_seg).to(predicted_noise_seg.device)
            beta_t = self._betas[t]
            sigma_t = beta_t.sqrt()
            data_seg = data_seg + sigma_t * z

        return data_im, data_seg

    def inference_call_alt(self, xt: torch.Tensor, predicted_noise: torch.Tensor): ...

    def prepare_betas(self):
        raise NotImplementedError(
            "Do not instantiate the base diffuser!!!!! Use a subclass instead"
        )


class LinearDiffuser(Diffuser):
    def prepare_betas(self):
        return torch.linspace(self.min_beta, self.max_beta, self.timesteps)


class CosDiffuser(Diffuser):

    def prepare_betas(self, s=0.008):
        def f(t):
            return torch.cos((t / self.timesteps + s) / (1 + s) * 0.5 * torch.pi) ** 2

        x = torch.linspace(0, self.timesteps, self.timesteps + 1)
        alphas_cumulative_prod = f(x) / f(torch.tensor([0]))
        betas = 1 - alphas_cumulative_prod[1:] / alphas_cumulative_prod[:-1]
        betas = torch.clip(betas, self.min_beta, self.max_beta)
        return betas
