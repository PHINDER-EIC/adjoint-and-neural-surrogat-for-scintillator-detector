from dataclasses import dataclass, field, asdict
from typing import Optional, Mapping, Any
import math
import json


@dataclass(frozen=True)
class Material:
    """
    Examples
    ---
    >>> AIR = Material("air", refractive_index=1.0, absorption_mu=0.0)
    >>> CAL = Material("scintillator", refractive_index=1.58, absorption_mu=0.01)
    >>> FIBER_CORE = Material("fiber_core", refractive_index=1.59, absorption_mu=0.005)
    >>> FIBER_CLAD = Material("fiber_cladding", refractive_index=1.49, absorption_mu=0.002)
    """

    name: str
    refractive_index: float
    absorption_mu: float = 0.0


@dataclass(frozen=True)
class ReflectiveMaterial(Material):
    reflectance: float = 1.0
    transmittance: float = 0.0
    specular_fraction:float = 1
    absorption_mu = None
    refractive_index = None


@dataclass(frozen=True)
class PlasticScintillator(Material):
    """
    Plastic scintillator material.

    Units
    -----
    refractive_index: dimensionless
    absorption_mu: 1/cm
    density_g_cm3: g/cm^3
    light_yield_photons_per_MeV: photons / MeV
    light_output_relative_anthracene: percent
    emission_peak_nm: nm
    attenuation_length_cm: cm
    rise_time_ns: ns
    decay_time_ns: ns
    pulse_width_fwhm_ns: ns
    mean_free_path_cm: cm
    softening_point_C: °C
    linear_expansion_coeff_C_inv: 1/°C
    """

    # ====== Scintillation output ======
    light_yield_photons_per_MeV: Optional[float] = None
    light_output_relative_anthracene: Optional[float] = None

    # ====== Optical properties ======
    emission_peak_nm: Optional[float] = None
    attenuation_length_cm: Optional[float] = None
    mean_free_path_cm: Optional[float] = None

    # Optional spectra: {wavelength_nm: value}
    emission_spectrum: Optional[Mapping[float, float]] = None
    refractive_index_spectrum: Optional[Mapping[float, float]] = None
    absorption_spectrum: Optional[Mapping[float, float]] = None

    # ====== Timing properties ======
    rise_time_ns: Optional[float] = None
    decay_time_ns: Optional[float] = None
    pulse_width_fwhm_ns: Optional[float] = None

    # ====== Material / mechanical properties ======
    density_g_cm3: Optional[float] = None
    polymer_base: str = "polyvinyltoluene"
    softening_point_C: Optional[float] = None
    linear_expansion_coeff_C_inv: Optional[float] = None

    # ====== Composition ======
    hydrogen_atoms_per_cm3: Optional[float] = None
    carbon_atoms_per_cm3: Optional[float] = None
    electrons_per_cm3: Optional[float] = None

    # ====== Quenching / detector-response parameters ======
    birks_constant_mm_MeV: Optional[float] = None

    # ====== Metadata ======
    manufacturer: Optional[str] = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self._validate_positive("refractive_index", self.refractive_index)
        self._validate_non_negative("absorption_mu", self.absorption_mu)

        optional_positive_fields = {
            "light_yield_photons_per_MeV": self.light_yield_photons_per_MeV,
            "light_output_relative_anthracene": self.light_output_relative_anthracene,
            "emission_peak_nm": self.emission_peak_nm,
            "attenuation_length_cm": self.attenuation_length_cm,
            "mean_free_path_cm": self.mean_free_path_cm,
            "rise_time_ns": self.rise_time_ns,
            "decay_time_ns": self.decay_time_ns,
            "pulse_width_fwhm_ns": self.pulse_width_fwhm_ns,
            "density_g_cm3": self.density_g_cm3,
            "softening_point_C": self.softening_point_C,
            "hydrogen_atoms_per_cm3": self.hydrogen_atoms_per_cm3,
            "carbon_atoms_per_cm3": self.carbon_atoms_per_cm3,
            "electrons_per_cm3": self.electrons_per_cm3,
            "birks_constant_mm_MeV": self.birks_constant_mm_MeV,
        }

        for field_name, value in optional_positive_fields.items():
            if value is not None:
                self._validate_positive(field_name, value)

        if self.linear_expansion_coeff_C_inv is not None:
            self._validate_non_negative(
                "linear_expansion_coeff_C_inv",
                self.linear_expansion_coeff_C_inv,
            )

    @staticmethod
    def _validate_positive(name: str, value: float) -> None:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number, got {value!r}")

    @staticmethod
    def _validate_non_negative(name: str, value: float) -> None:
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"{name} must be a non-negative finite number, got {value!r}"
            )

    @property
    def absorption_length_cm(self) -> float:
        """
        Returns the absorption length corresponding to absorption_mu.

        If absorption_mu = 0, the material is treated as non-absorbing.
        """
        if self.absorption_mu == 0:
            return math.inf
        return 1.0 / self.absorption_mu

    def expected_photons(self, deposited_energy_MeV: float) -> Optional[float]:
        """
        Expected scintillation photons for a given deposited energy.

        Returns None if light_yield_photons_per_MeV is not defined.
        """
        self._validate_non_negative("deposited_energy_MeV", deposited_energy_MeV)

        if self.light_yield_photons_per_MeV is None:
            return None

        return deposited_energy_MeV * self.light_yield_photons_per_MeV

    def transmitted_fraction(self, path_length_cm: float) -> float:
        """
        Beer-Lambert attenuation through the material.

        T = exp(-mu * x)
        """
        self._validate_non_negative("path_length_cm", path_length_cm)
        return math.exp(-self.absorption_mu * path_length_cm)

    def critical_angle_deg(
        self, external_refractive_index: float = 1.0
    ) -> Optional[float]:
        """
        Critical angle for total internal reflection at an interface.

        Returns None if total internal reflection is not possible.
        """
        self._validate_positive("external_refractive_index", external_refractive_index)

        if self.refractive_index <= external_refractive_index:
            return None

        theta_rad = math.asin(external_refractive_index / self.refractive_index)
        return math.degrees(theta_rad)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 4) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def ej228(cls, absorption_mu: float = 0.01) -> "PlasticScintillator":
        """
        Factory for Eljen EJ-228.

        absorption_mu is left user-configurable because in simulations it often
        depends on wavelength, surface treatment, geometry, and calibration.
        """
        return cls(
            name="EJ-228",
            manufacturer="Eljen Technology",
            refractive_index=1.58,
            absorption_mu=absorption_mu,
            light_output_relative_anthracene=67.0,
            light_yield_photons_per_MeV=10200.0,
            emission_peak_nm=391.0,
            mean_free_path_cm=100.0,
            rise_time_ns=0.5,
            decay_time_ns=1.4,
            pulse_width_fwhm_ns=1.2,
            density_g_cm3=1.023,
            polymer_base="polyvinyltoluene",
            softening_point_C=75.0,
            linear_expansion_coeff_C_inv=7.8e-5,
            hydrogen_atoms_per_cm3=5.15e22,
            carbon_atoms_per_cm3=4.69e22,
            electrons_per_cm3=3.33e23,
            notes=(
                "Fast timing plastic scintillator.",
                "Best used in small dimensions for timing applications.",
                "Light guides should be avoided when optimizing timing.",
            ),
        )

    @classmethod
    def ej200(cls, absorption_mu: float = 1.0 / 380.0) -> "PlasticScintillator":
        return cls(
            name="EJ-200",
            manufacturer="Eljen Technology",
            refractive_index=1.58,
            absorption_mu=absorption_mu,
            attenuation_length_cm=380.0,
            emission_peak_nm=425.0,
            rise_time_ns=0.9,
            decay_time_ns=2.1,
            light_output_relative_anthracene=64.0,
            light_yield_photons_per_MeV=10000.0,
            density_g_cm3=1.023,
            polymer_base="polyvinyltoluene",
        )

    @classmethod
    def ej204(cls, absorption_mu: float = 1.0 / 160.0) -> "PlasticScintillator":
        return cls(
            name="EJ-204",
            manufacturer="Eljen Technology",
            refractive_index=1.58,
            absorption_mu=absorption_mu,
            attenuation_length_cm=160.0,
            emission_peak_nm=408.0,
            rise_time_ns=0.7,
            decay_time_ns=1.8,
            light_output_relative_anthracene=68.0,
            light_yield_photons_per_MeV=10400.0,
            density_g_cm3=1.023,
            polymer_base="polyvinyltoluene",
        )

    @classmethod
    def ej230(cls, absorption_mu: float = 1.0 / 120.0) -> "PlasticScintillator":
        return cls(
            name="EJ-230",
            manufacturer="Eljen Technology",
            refractive_index=1.58,
            absorption_mu=absorption_mu,
            attenuation_length_cm=120.0,
            emission_peak_nm=391.0,
            rise_time_ns=0.5,
            decay_time_ns=1.5,
            light_output_relative_anthracene=64.0,
            light_yield_photons_per_MeV=9700.0,
            density_g_cm3=1.023,
            polymer_base="polyvinyltoluene",
        )

    @classmethod
    def ej260(cls, absorption_mu: float = 1.0 / 350.0) -> "PlasticScintillator":
        return cls(
            name="EJ-260",
            manufacturer="Eljen Technology",
            refractive_index=1.58,
            absorption_mu=absorption_mu,
            attenuation_length_cm=350.0,
            emission_peak_nm=490.0,
            rise_time_ns=1.5,
            decay_time_ns=9.2,
            light_output_relative_anthracene=60.0,
            light_yield_photons_per_MeV=9200.0,
            density_g_cm3=1.023,
            polymer_base="polyvinyltoluene",
        )

    @classmethod
    def bc408(cls, absorption_mu: float = 1.0 / 380.0) -> "PlasticScintillator":
        return cls(
            name="BC-408",
            manufacturer="Luxium / Saint-Gobain",
            refractive_index=1.58,
            absorption_mu=absorption_mu,
            attenuation_length_cm=380.0,
            emission_peak_nm=425.0,
            rise_time_ns=0.9,
            decay_time_ns=2.1,
            light_output_relative_anthracene=64.0,
            light_yield_photons_per_MeV=10000.0,
            density_g_cm3=1.023,
            polymer_base="polyvinyltoluene",
        )

    @classmethod
    def bc404(cls, absorption_mu: float = 1.0 / 160.0) -> "PlasticScintillator":
        return cls(
            name="BC-404",
            manufacturer="Luxium / Saint-Gobain",
            refractive_index=1.58,
            absorption_mu=absorption_mu,
            attenuation_length_cm=160.0,
            emission_peak_nm=408.0,
            rise_time_ns=0.7,
            decay_time_ns=1.8,
            light_output_relative_anthracene=68.0,
            light_yield_photons_per_MeV=10400.0,
            density_g_cm3=1.023,
            polymer_base="polyvinyltoluene",
        )

    @classmethod
    def bc420(cls, absorption_mu: float = 1.0 / 110.0) -> "PlasticScintillator":
        return cls(
            name="BC-420",
            manufacturer="Luxium / Saint-Gobain",
            refractive_index=1.58,
            absorption_mu=absorption_mu,
            attenuation_length_cm=110.0,
            emission_peak_nm=391.0,
            rise_time_ns=0.5,
            decay_time_ns=1.5,
            light_output_relative_anthracene=64.0,
            light_yield_photons_per_MeV=10000.0,
            density_g_cm3=1.023,
            polymer_base="polyvinyltoluene",
        )

    @classmethod
    def bc422(cls, absorption_mu: float = 1.0 / 8.0) -> "PlasticScintillator":
        return cls(
            name="BC-422",
            manufacturer="Luxium / Saint-Gobain",
            refractive_index=1.58,
            absorption_mu=absorption_mu,
            attenuation_length_cm=8.0,
            emission_peak_nm=370.0,
            rise_time_ns=0.35,
            decay_time_ns=1.6,
            light_output_relative_anthracene=55.0,
            light_yield_photons_per_MeV=8500.0,
            density_g_cm3=1.023,
            polymer_base="polyvinyltoluene",
        )

    @classmethod
    def scsn81(cls, absorption_mu: float = 1.0 / 250.0) -> "PlasticScintillator":
        """
        Factory for Kuraray SCSN-81.
        Widely used in HEP calorimetry (CMS HCAL, etc.).
        """

        return cls(
            name="SCSN-81",
            manufacturer="Kuraray",
            refractive_index=1.59,
            absorption_mu=absorption_mu,
            light_output_relative_anthracene=78.0,
            light_yield_photons_per_MeV=12000.0,
            emission_peak_nm=440.0,
            attenuation_length_cm=250.0,
            rise_time_ns=0.9,
            decay_time_ns=2.4,
            density_g_cm3=1.032,
            polymer_base="polystyrene",
            hydrogen_atoms_per_cm3=5.2e22,
            carbon_atoms_per_cm3=4.8e22,
            electrons_per_cm3=3.4e23,
            notes=(
                "Plastic scintillator developed for high-energy physics calorimetry.",
                "Used in CMS HCAL and similar detector systems.",
                "Optimized for wavelength-shifting fiber readout.",
            ),
        )
