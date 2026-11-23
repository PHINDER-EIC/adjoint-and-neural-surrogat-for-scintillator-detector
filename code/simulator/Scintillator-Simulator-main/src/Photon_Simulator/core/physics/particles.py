from abc import ABC


class Particle(ABC):
    def __init__(
        self,
        name: str,
        pdg_id: int,
        mass: float,        # [MeV/c^2]
        charge: int,        # in units of e
        spin: float,
        is_mip_candidate: bool = False,
    ):
        self.name = name
        self.pdg_id = pdg_id
        self.mass = mass
        self.charge = charge
        self.spin = spin
        self.is_mip_candidate = is_mip_candidate

        self.energy = None      # total energy [MeV]
        self.momentum = None    # 3-vector [MeV/c]
        self.beta = None
        self.gamma = None

    def set_kinematics(self, energy: float, momentum: list[float]):
        self.energy = energy
        self.momentum = momentum
        self.gamma = energy / self.mass
        self.beta = (1.0 - 1.0 / self.gamma**2) ** 0.5
        


class MuonMinus(Particle):
    def __init__(self):
        super().__init__(
            name="mu-",
            pdg_id=13,
            mass=105.658,
            charge=-1,
            spin=0.5,
            is_mip_candidate=True,
        )



class MIP(Particle):
    """
    Minimum Ionizing Particle.

    Useful as an effective particle for calorimeter calibration.
    Typically behaves like a relativistic charged particle
    with approximately minimum dE/dx.
    >>> mip = MIP(mu)
    >>> mip.set_kinematics(energy=1000.0, momentum=[0.0, 0.0, 994.4])
    """

    def __init__(self, reference_particle: Particle | None = None):
        if reference_particle is None:
            reference_particle = MuonMinus()

        super().__init__(
            name=f"MIP({reference_particle.name})",
            pdg_id=reference_particle.pdg_id,
            mass=reference_particle.mass,
            charge=reference_particle.charge,
            spin=reference_particle.spin,
            is_mip_candidate=True,
        )

        self.reference_particle = reference_particle