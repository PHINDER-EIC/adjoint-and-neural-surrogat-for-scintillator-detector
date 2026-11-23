import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (torch.norm(v, dim=-1, keepdim=True) + eps)


def reflect(direction: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    """
    The idea is that reflected vector is r = d - 2(d ⋅ n) n
    Reflect incident direction around surface normal.
    direction: [..., 3] 
    normal:    [..., 3]
    In this way I can work with any tensor, but the last dimenson has to be 3.     
    """
    direction = normalize(direction)
    normal = normalize(normal)
    #? keepdim = True to maintain the shape [...,1]
    return normalize(direction - 2.0 * torch.sum(direction * normal, dim=-1, keepdim=True) * normal)


def refract(direction: torch.Tensor, normal: torch.Tensor, n1: torch.Tensor, n2: torch.Tensor,
            eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable refraction using Snell's law in vector form.
    eps is just to avoid zero division. 
    Returns:
        refracted_dir: [..., 3]
        trasmission_check:      [..., 1]   1 if total internal reflection, else 0
        
    TODO:
    - Try to eliminate eps
    """
    direction = normalize(direction)
    normal = normalize(normal)

    cos_i = -torch.sum(direction * normal, dim=-1, keepdim=True).clamp(-1.0, 1.0)
    eta = n1 / (n2 + eps)

    sin_t2 = eta**2 * (1.0 - cos_i**2)
    #NOTE to avoid broken model due to sqrt of negative number
    trasmission_check = (sin_t2 > 1.0).float()

    cos_t = torch.sqrt(torch.clamp(1.0 - sin_t2, min=0.0))
    refracted = eta * direction + (eta * cos_i - cos_t) * normal
    refracted = normalize(refracted)

    return refracted, trasmission_check

def fresnel_coefficients(
    direction: torch.Tensor,
    normal: torch.Tensor,
    n1: torch.Tensor,
    n2: torch.Tensor,
    polarization: str = "unpolarized",
    eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Fresnel reflection and transmission coefficients.

    Parameters
    ----------
    direction : torch.Tensor
        Incident ray direction, shape [..., 3].
    normal : torch.Tensor
        Surface normal, shape [..., 3].
    n1 : torch.Tensor
        Refractive index of the incident medium, broadcastable to [..., 1].
    n2 : torch.Tensor
        Refractive index of the transmitted medium, broadcastable to [..., 1].
    polarization : str
        "s", "p", or "unpolarized".
    eps : float
        Small value for numerical stability.

    Returns
    -------
    R : torch.Tensor
        Reflection coefficient, shape [..., 1].
    T : torch.Tensor
        Transmission coefficient, shape [..., 1].

    Notes
    -----
    - Handles total internal reflection.
    - Assumes non-absorbing media.
    - Returns energy coefficients.
    - Absoption is managed outside
    """

    direction = normalize(direction)
    normal = normalize(normal)

    # cos(theta_i) with the convention that direction points toward the surface
    cos_i = torch.sum(-direction * normal, dim=-1, keepdim=True).clamp(-1.0, 1.0)
    cos_i = torch.abs(cos_i)

    # Relative refractive index
    eta = n1 / (n2 + eps)

    # Snell's law: sin^2(theta_t) = eta^2 * (1 - cos^2(theta_i))
    sin_t2 = eta**2 * (1.0 - cos_i**2)

    # Total internal reflection mask
    tir = sin_t2 > 1.0

    # cos(theta_t), blocked for numerical stability
    cos_t = torch.sqrt(torch.clamp(1.0 - sin_t2, min=0.0))

    # Fresnel reflectance for s polarization
    Rs = ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t +eps))**2

    # Fresnel reflectance for p polarization
    Rp = ((n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t+ eps))**2

    # In total internal reflection, all energy is reflected
    Rs = torch.where(tir, torch.ones_like(Rs), Rs)
    Rp = torch.where(tir, torch.ones_like(Rp), Rp)

    if polarization == "s":
        R = Rs
    elif polarization == "p":
        R = Rp
    elif polarization == "unpolarized":
        R = 0.5 * (Rs + Rp)
    else:
        raise ValueError("polarization must be 's', 'p', or 'unpolarized'")

    T = 1.0 - R
    return R, T


def absorption_coeff(absorption_logit: torch.Tensor) -> torch.Tensor:
    """
    Converts an unconstrained parameter into [0, 1].
    !FROM AI - TO CHECK
    """
    return torch.sigmoid(absorption_logit)

def beer_lambert_absorption(alpha: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    """
    A = 1 - exp(-alpha * distance)
    alpha:    [..., 1]
    distance: [..., 1]
    """
    return 1.0 - torch.exp(-alpha * distance)


class DifferentiablePhotonStep(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        pos: torch.Tensor,         # [N, 3]
        direction: torch.Tensor,   # [N, 3]
        weight: torch.Tensor,      # [N, 1]
        normal: torch.Tensor,      # [N, 3]
        n1: torch.Tensor,          # [N, 1]
        n2: torch.Tensor,          # [N, 1]
        distance: torch.Tensor,    # [N, 1] distance to next interaction
        absorption_alpha: torch.Tensor  # [N, 1]
    ):
        direction = normalize(direction)
        normal = normalize(normal)

        # Move photon to interaction point
        hit_pos = pos + distance * direction

        # Fresnel reflection/transmission
        R, T = fresnel_coefficients(direction, normal, n1, n2)

        # Absorption in medium before/at interaction
        A = beer_lambert_absorption(absorption_alpha, distance)

        # Ensure reflected/transmitted part is scaled by non-absorbed fraction
        survive = 1.0 - A
        R_eff = survive * R
        T_eff = survive * T

        # Reflected and refracted states
        dir_reflect = reflect(direction, normal)
        dir_refract, tir_mask = refract(direction, normal, n1, n2)

        # If total internal reflection happens, transmission goes to reflection
        T_eff = T_eff * (1.0 - tir_mask)
        R_eff = R_eff + survive * T * tir_mask

        # Weighted average state
        next_dir = R_eff * dir_reflect + T_eff * dir_refract
        next_dir = normalize(next_dir)

        # Total surviving weight
        next_weight = weight * (R_eff + T_eff)

        # Position: after the interaction we keep the hit point
        next_pos = hit_pos

        return next_pos, next_dir, next_weight, {
            "R": R,
            "T": T,
            "A": A,
            "R_eff": R_eff,
            "T_eff": T_eff,
            "tir_mask": tir_mask
        }