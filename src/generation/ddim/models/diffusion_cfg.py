"""
Covariate-conditioned DDIM denoiser for classifier-free guidance.

Same residual-MLP backbone as ModelDDIM (ResLinear blocks, Lacan et al.), but the
conditioning vector c is built from one learned encoder per covariate:

    c = concat( E_cancer(cancer), E_sex(sex), E_race(race), E_age(age) )

  * cancer / sex / race: nn.Embedding with one extra row for the CFG null token.
  * age: continuous; normalised age -> small MLP. A learned vector stands in
    when age is null (NaN).

c is concatenated into every ResLinear block exactly where the baseline
concatenates its tissue one-hot.
"""
import torch
import torch.nn as nn

from src.generation.ddim.models.diffusion_ddim import (ResLinear, Normalize, nonlinearity,
                                                       get_timestep_embedding)
from src.generation.ddim.covariates import N_CANCER, N_SEX, N_RACE, COLS


class CovariateEncoder(nn.Module):
    def __init__(self, dim_cancer=16, dim_sex=4, dim_race=8, dim_age=8):
        super().__init__()
        # +1 row = null token
        self.cancer = nn.Embedding(N_CANCER + 1, dim_cancer)
        self.sex = nn.Embedding(N_SEX + 1, dim_sex)
        self.race = nn.Embedding(N_RACE + 1, dim_race)
        self.age = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, dim_age))
        self.age_null = nn.Parameter(torch.zeros(dim_age))
        self.out_dim = dim_cancer + dim_sex + dim_race + dim_age

    def forward(self, y):
        y = y.float()
        cancer = self.cancer(y[:, COLS["cancer"]].long())
        sex = self.sex(y[:, COLS["sex"]].long())
        race = self.race(y[:, COLS["race"]].long())

        age_raw = y[:, COLS["age"]:COLS["age"] + 1]
        age_null = torch.isnan(age_raw)
        age = self.age(torch.nan_to_num(age_raw, nan=0.0))
        age = torch.where(age_null, self.age_null.to(age.dtype).expand_as(age), age)

        return torch.cat([cancer, sex, race, age], dim=1)


class ModelDDIMCFG(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        m = config.model
        cov = getattr(config, "cfg", None)

        dropout = m.dropout
        self.dim_t = m.dim_t
        self.temb_ch = self.dim_t * 4
        self.is_time_embed = m.is_time_embed
        self.resolution = config.data.image_size
        self.d_layers = m.d_layers
        self.timestep_max = config.diffusion.num_diffusion_timesteps

        self.cov_encoder = CovariateEncoder(
            dim_cancer=getattr(cov, "dim_cancer", 16),
            dim_sex=getattr(cov, "dim_sex", 4),
            dim_race=getattr(cov, "dim_race", 8),
            dim_age=getattr(cov, "dim_age", 8),
        )
        cond_dim = self.cov_encoder.out_dim

        if self.is_time_embed:
            self.temb_dense = nn.ModuleList([
                nn.Linear(self.temb_ch, self.temb_ch),
                nn.Linear(self.temb_ch, self.dim_t),
            ])

        self.mid = nn.ModuleList([
            ResLinear(
                in_channels=self.d_layers[i - 1] if i else self.resolution,
                out_channels=self.d_layers[i],
                temb_channels=self.temb_ch,
                dropout=dropout,
                num_classes=cond_dim,
                float_precision=m.precision,
            )
            for i in range(len(self.d_layers))
        ])
        self.norm_out = Normalize(self.d_layers[-1])
        self.lin_out = nn.Linear(self.d_layers[-1], self.resolution)

    def forward(self, x, t, y):
        if self.is_time_embed:
            temb = get_timestep_embedding(t, self.temb_ch)
            temb = self.temb_dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb_dense[1](temb)
        else:
            temb = torch.unsqueeze(t / self.timestep_max, dim=1)

        c = nonlinearity(self.cov_encoder(y))

        h = x
        for block in self.mid:
            h = block(h, temb, c)

        h = self.norm_out(h)
        h = nonlinearity(h)
        return self.lin_out(h)


class CFGWrapper(nn.Module):
    """
    Wraps a trained ModelDDIMCFG so it can be dropped into the existing samplers
    (generalized_steps / ddpm_steps call model(x, t, y)).

    Given guidance weights {covariate: w}, the noise estimate is

        eps = eps(x, y_base) + sum_k  w_k * ( eps(x, y_base + k) - eps(x, y_base) )

    where y_base is y with every guided covariate nulled, and "y_base + k" restores
    only covariate k. Unguided covariates stay as given in y (i.e. are plain
    conditioning). With a single guided covariate this is standard CFG:
    eps_uncond + w * (eps_cond - eps_uncond); w = 1 is ordinary conditional
    sampling, w = 0 ignores that covariate, w > 1 exaggerates it.

    All forward passes are batched into one call.
    """
    def __init__(self, model, weights: dict):
        super().__init__()
        from src.generation.ddim.covariates import null_out
        self.model = model
        self.weights = {k: float(v) for k, v in (weights or {}).items()}
        for k in self.weights:
            assert k in COLS, f"unknown covariate {k}"
        self._null_out = null_out

    def forward(self, x, t, y):
        if not self.weights:
            return self.model(x, t, y)
        names = list(self.weights)
        y_base = self._null_out(y, names)
        ys = [y_base]
        for k in names:
            yk = y_base.clone()
            yk[:, COLS[k]] = y[:, COLS[k]]
            ys.append(yk)
        n_pass = len(ys)
        eps = self.model(x.repeat(n_pass, 1), t.repeat(n_pass), torch.cat(ys, dim=0))
        eps = eps.view(n_pass, x.shape[0], -1)
        out = eps[0]
        for i, k in enumerate(names, start=1):
            out = out + self.weights[k] * (eps[i] - eps[0])
        return out
