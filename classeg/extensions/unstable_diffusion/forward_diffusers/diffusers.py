import torch
import warnings
from torch.utils.data import WeightedRandomSampler

UNIFORM = "uniform"
PRIORTY = "priority"

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
        self._max_t_to_sample = self.timesteps
        self._t_sample_style = UNIFORM

    @property
    def max_t_to_sample(self):
        return self._max_t_to_sample

    def __call__(self, im: torch.Tensor, seg: torch.Tensor, t=None, noise_im=None, noise_seg=None):
        """
        Given data, randomly samples timesteps and return noise, noisy_data, timesteps.
        Keeps devices uniform.
        """
        if t is None:
            if self._t_sample_style == UNIFORM:
                weights = [1 for _ in range(0, self._max_t_to_sample)]
                weights[0] = 0
            elif self._t_sample_style == PRIORTY:
                timestep_sum = (self.timesteps * (self.timesteps + 1)) / 2
                weights = [i/timestep_sum for i in range(0, self._max_t_to_sample)]

            t = torch.tensor(list(WeightedRandomSampler(weights, im.shape[0], replacement=True))).long()
            # t = torch.randint(1, self._max_t_to_sample, (,)).long()
        if noise_im is None:
            noise_im = torch.randn_like(im).to(im.device)
        if noise_seg is None:
            noise_seg = torch.randn_like(seg).to(im.device)
            # print(noise_seg.shape)

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

    def set_max_t(self, max_t):
        self._max_t_to_sample = max_t

    def inference_call(
        self, 
        im: torch.Tensor, 
        seg:torch.Tensor,
        predicted_noise_im: torch.Tensor, 
        predicted_noise_seg: torch.Tensor, 
        t: int, clamp=False, training_time=False,**kwargs
    ):
        """
        For use in inference mode
        If clamp is true, clamps data between -1 and 1
        """
        if isinstance(t, int):
            t = torch.tensor([t]).to(im.device)
        alpha_t = self._alphas.to(im.device)[t].to(im.device)
        alpha_t_bar = self._alpha_bars.to(im.device)[t].to(im.device)

        data_im = (1 / alpha_t.sqrt()).reshape(-1, 1, 1, 1) * (
            im - ((1 - alpha_t) / (1 - alpha_t_bar).sqrt()).reshape(-1, 1, 1, 1) * predicted_noise_im
        )

        if clamp:
            data_im = torch.clip(data_im, -5, 5)
        if not training_time and t > 0:
            z = torch.randn_like(predicted_noise_im).to(predicted_noise_im.device)
            beta_t = self._betas.to(im.device)[t]
            sigma_t = beta_t.sqrt()
            data_im = data_im + sigma_t * z

        # SEG
        data_seg = (1 / alpha_t.sqrt()).reshape(-1, 1, 1, 1) * (
            seg - ((1 - alpha_t) / (1 - alpha_t_bar).sqrt()).reshape(-1, 1, 1, 1) * predicted_noise_seg
        )

        if clamp:
            data_seg = torch.clip(data_seg, -5, 5)
        if not training_time and t > 0:
            z = torch.randn_like(predicted_noise_seg).to(predicted_noise_seg.device)
            beta_t = self._betas.to(im.device)[t]
            sigma_t = beta_t.sqrt()
            data_seg = data_seg + sigma_t * z

        return data_im, data_seg

    def inference_call_alt(self, xt: torch.Tensor, predicted_noise: torch.Tensor): ...

    def prepare_betas(self):
        raise NotImplementedError(
            "Do not instantiate the base diffuser!!!!! Use a subclass instead"
        )


class DDIMDiffuser(Diffuser):
    def prepare_betas(self):
        raise Exception("DDIM Diffuser should not be used. Override it with linear or cosine diffuser")
    
    def inference_call(self, 
                       im: torch.Tensor, 
                       seg: torch.Tensor, 
                       predicted_noise_im: torch.Tensor, 
                       predicted_noise_seg: torch.Tensor, 
                       t: int, i=0, **kwargs):
        """
        For use in inference mode
        If clamp is true, clamps data between -1 and 1
        """
        if isinstance(t, int):
            t = torch.tensor([t]).to(im.device)

        # alphas = self._alphas.to(im.device)
        alpha_bars = self._alpha_bars.to(im.device)
        
        alpha_t_bar = alpha_bars[-i]
        alpha_tm1_bar = alpha_bars[-i-1]


        c1 = 0 * ((1 - alpha_t_bar / alpha_tm1_bar) * (1 - alpha_t_bar) / (1 - alpha_t_bar)).sqrt()
        c2 = ((1-alpha_tm1_bar) - c1 ** 2).sqrt()

        im0_t = (im-predicted_noise_im*(1-alpha_t_bar).sqrt())/alpha_t_bar.sqrt()
        data_im = alpha_tm1_bar.sqrt()*im0_t + c2*predicted_noise_im + c1*torch.randn_like(im)

        seg0_t = (seg- predicted_noise_seg*(1-alpha_t_bar).sqrt() )/(alpha_t_bar.sqrt())
        data_seg = alpha_tm1_bar.sqrt()*seg0_t + c2*predicted_noise_seg+ c1*torch.randn_like(seg)

        return data_im, data_seg

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

class LinearDDIM(LinearDiffuser, DDIMDiffuser):
    def prepare_betas(self):
        return torch.cat([torch.zeros(1), torch.linspace(self.min_beta, self.max_beta, self.timesteps)], dim=0)

class CosDDIM(CosDiffuser, DDIMDiffuser):
    def prepare_betas(self, s=0.008):
        def f(t):
            return torch.cos((t / self.timesteps + s) / (1 + s) * 0.5 * torch.pi) ** 2

        x = torch.linspace(0, self.timesteps, self.timesteps + 1)
        alphas_cumulative_prod = f(x) / f(torch.tensor([0]))
        betas = 1 - alphas_cumulative_prod[1:] / alphas_cumulative_prod[:-1]
        betas = torch.clip(betas, self.min_beta, self.max_beta)
        return torch.cat([torch.zeros(1), betas],dim=1)
