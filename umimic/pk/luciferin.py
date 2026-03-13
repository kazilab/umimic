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

    Parameters:
        dose: Luciferin dose (mg/kg or ug/well)
        ka_luc: Absorption rate constant (1/min)
        ke_luc: Elimination rate constant (1/min)
        km: Michaelis-Menten constant for luciferase reaction
    """

    dose: float = 150.0  # mg/kg (standard IP dose)
    ka_luc: float = 0.5  # 1/min
    ke_luc: float = 0.05  # 1/min (slow clearance)
    km: float = 50.0  # substrate units

    @property
    def peak_time(self) -> float:
        """Time of peak luciferin concentration (minutes post-injection)."""
        if self.ka_luc <= self.ke_luc:
            return 0.0
        return np.log(self.ka_luc / self.ke_luc) / (self.ka_luc - self.ke_luc)

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

    Attenuation model: Att = exp(-mu_eff * depth)
    where mu_eff is the effective attenuation coefficient.
    """

    mu_eff: float = 0.5  # mm^-1, effective attenuation coefficient
    reference_depth: float = 2.0  # mm, reference tumor depth

    def attenuation_factor(
        self, depth: float | None = None, volume: float | None = None
    ) -> float:
        """Compute attenuation factor (0 to 1).

        Args:
            depth: Tumor depth in mm (if known).
            volume: Tumor volume in mm^3 (used to estimate depth if depth unknown).

        Returns:
            Fraction of photons reaching detector (1.0 = no attenuation).
        """
        if depth is None:
            if volume is not None:
                # Estimate depth from volume (sphere approximation)
                radius = (3 * volume / (4 * np.pi)) ** (1 / 3)
                depth = self.reference_depth + radius
            else:
                depth = self.reference_depth

        return float(np.exp(-self.mu_eff * depth))
