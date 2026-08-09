"""Luciferin substrate kinetics for bioluminescence imaging.

Models the availability of luciferin after injection, which affects
the BLI signal independently of cell number. Failure to account for
luciferin kinetics can cause misinterpretation of signal changes as
cell death when they are actually substrate-related.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LuciferinKinetics:
    """Model of luciferin bioavailability after IP/IV injection.

    The luciferin concentration follows a one-compartment model:
        C_luc(t) = dose * ka / (ka - ke) * (exp(-ke*t) - exp(-ka*t))

    The luciferase reaction rate saturates with substrate:
        g(C_luc) = C_luc / (Km + C_luc)

    .. note::
       **These units are phenomenological, not biochemical.** `dose` is an
       administered amount (mg/kg) while `km` is an intracellular substrate
       concentration; the model omits the volume of distribution and the
       tissue-uptake step that would connect them, so `C_luc` is in arbitrary
       units and `dose / km` merely sets where on the saturation curve the
       imaging window falls. The defaults place a standard 150 mg/kg IP dose
       comfortably above `km`, i.e. near saturation.

       Use this to model the *shape* of the signal-versus-imaging-time curve
       and the relative penalty for imaging off-peak. Do not read `km` as a
       measured Michaelis constant, and do not compare `C_luc` against
       published luciferin concentrations. :meth:`signal_fraction` is
       normalized to the peak precisely so that only relative timing matters.

    Parameters:
        dose: Luciferin dose (mg/kg or ug/well), arbitrary units
        ka_luc: Absorption rate constant (1/min)
        ke_luc: Elimination rate constant (1/min)
        km: Half-saturation constant, in the same arbitrary units as C_luc
    """

    dose: float = 150.0  # mg/kg (standard IP dose)
    ka_luc: float = 0.5  # 1/min
    ke_luc: float = 0.05  # 1/min (slow clearance)
    km: float = 50.0  # substrate units

    def __post_init__(self) -> None:
        if not np.isfinite(self.dose) or self.dose < 0:
            raise ValueError(f"dose must be finite and non-negative, got {self.dose}.")
        if not np.isfinite(self.ka_luc) or self.ka_luc <= 0:
            raise ValueError(
                f"ka_luc must be a positive rate constant, got {self.ka_luc}."
            )
        if not np.isfinite(self.ke_luc) or self.ke_luc <= 0:
            raise ValueError(
                f"ke_luc must be a positive rate constant, got {self.ke_luc}."
            )
        if not np.isfinite(self.km) or self.km <= 0:
            raise ValueError(f"km must be a positive constant, got {self.km}.")

    @property
    def peak_time(self) -> float:
        """Time of peak luciferin concentration (minutes post-injection).

        t_max = ln(ka/ke) / (ka - ke) is valid for both ka > ke and ka < ke:
        when ka < ke both the numerator and denominator change sign, so the
        result stays positive. Only ka == ke needs the limiting form
        t_max = 1/ka.
        """
        ka, ke = self.ka_luc, self.ke_luc
        if abs(ka - ke) < 1e-10:
            return float(1.0 / ka)
        return float(np.log(ka / ke) / (ka - ke))

    def substrate_concentration(self, t_post_injection: float) -> float:
        """Luciferin concentration at time t after injection (minutes).

        Args:
            t_post_injection: Time since luciferin injection (minutes).
        """
        t = max(t_post_injection, 0.0)
        ka, ke = self.ka_luc, self.ke_luc

        if abs(ka - ke) < 1e-10:
            # Limit case: ka ≈ ke
            return self.dose * ka * t * np.exp(-ka * t)

        return (
            self.dose * ka / (ka - ke) * (np.exp(-ke * t) - np.exp(-ka * t))
        )

    def signal_fraction(self, t_post_injection: float) -> float:
        """Fraction of maximum signal at time t.

        The enzymatic reaction rate follows Michaelis-Menten kinetics:
        g(C_luc) = C_luc / (Km + C_luc)

        Normalized so peak = 1.0.
        """
        c = self.substrate_concentration(t_post_injection)
        g = c / (self.km + c) if (self.km + c) > 0 else 0.0

        # Normalize to peak
        c_peak = self.substrate_concentration(self.peak_time)
        g_peak = c_peak / (self.km + c_peak) if (self.km + c_peak) > 0 else 1.0

        return g / g_peak if g_peak > 0 else 0.0

    def optimal_imaging_window(self, tolerance: float = 0.9) -> tuple[float, float]:
        """Find the time window where signal is within tolerance of peak.

        Args:
            tolerance: Fraction of peak signal to define the window.

        Returns:
            (t_start, t_end) in minutes where signal >= tolerance * peak.
        """
        times = np.linspace(0, self.peak_time * 4, 1000)
        fractions = np.array([self.signal_fraction(t) for t in times])

        above = np.where(fractions >= tolerance)[0]
        if len(above) == 0:
            return (self.peak_time, self.peak_time)

        return (float(times[above[0]]), float(times[above[-1]]))


@dataclass
class TissueAttenuation:
    """Model of optical attenuation in tissue for in vivo BLI.

    Photons are absorbed and scattered in tissue, causing signal loss
    that increases with tumor depth and size. Failure to model this
    can cause overestimation of cell death in growing tumors.

    Attenuation model: Att = exp(-mu_eff * depth) for a point source at
    `depth`. For a tumour of finite size the signal is emitted throughout the
    volume, so the attenuation is averaged over the emitting volume rather
    than evaluated at its centre -- see :meth:`attenuation_factor`.

    `reference_depth` is the depth of tissue *above* the tumour (skin and
    overlying tissue), not the depth of its centre.
    """

    mu_eff: float = 0.5  # mm^-1, effective attenuation coefficient
    reference_depth: float = 2.0  # mm, tissue depth above the tumour

    def __post_init__(self) -> None:
        if not np.isfinite(self.mu_eff) or self.mu_eff < 0:
            raise ValueError(
                f"mu_eff must be finite and non-negative, got {self.mu_eff}."
            )
        if not np.isfinite(self.reference_depth) or self.reference_depth < 0:
            raise ValueError(
                f"reference_depth must be finite and non-negative, "
                f"got {self.reference_depth}."
            )

    def attenuation_factor(
        self, depth: float | None = None, volume: float | None = None
    ) -> float:
        """Compute attenuation factor (0 to 1).

        With a `volume`, the tumour is treated as a sphere of radius r whose
        top lies at `reference_depth`, and the attenuation is averaged over
        the sphere:

            <Att> = exp(-mu*(d0 + r)) * 3*(x*cosh(x) - sinh(x)) / x**3,
            x = mu*r

        Evaluating ``exp(-mu * (d0 + r))`` alone -- the attenuation at the
        centroid -- systematically overstates signal loss, because exp is
        convex and the cells nearest the surface dominate the measured signal.
        That error grows with tumour size, so it can masquerade as progressive
        cell loss in a tumour that is actually growing.

        Args:
            depth: Depth of a point source in mm. Takes precedence over volume.
            volume: Tumor volume in mm^3.

        Returns:
            Fraction of photons reaching detector (1.0 = no attenuation).
        """
        if depth is not None:
            return float(np.exp(-self.mu_eff * max(float(depth), 0.0)))

        if volume is None or volume <= 0:
            return float(np.exp(-self.mu_eff * self.reference_depth))

        radius = (3.0 * float(volume) / (4.0 * np.pi)) ** (1.0 / 3.0)
        x = self.mu_eff * radius

        # Fold the exp(-mu*r) from the centre depth into the shape factor:
        #   exp(-x)*cosh(x) = (1 + e^-2x)/2,  exp(-x)*sinh(x) = (1 - e^-2x)/2
        # so the product stays bounded instead of multiplying an overflowing
        # cosh by an underflowing exponential.
        if x < 1e-2:
            # The closed form subtracts two nearly equal O(x) quantities to
            # produce an O(x^3) result, losing ~9 digits by x ~ 1e-5. Use the
            # series instead: exp(-x) * (1 + x^2/10 + ...) = 1 - x + 3x^2/5 ...
            shape_factor = 1.0 - x + 0.6 * x**2 - (4.0 / 15.0) * x**3
        else:
            e2 = np.exp(-2.0 * x)
            shape_factor = 3.0 * (x * (1.0 + e2) / 2.0 - (1.0 - e2) / 2.0) / x**3

        return float(np.exp(-self.mu_eff * self.reference_depth) * shape_factor)
