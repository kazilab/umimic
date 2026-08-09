r"""Dose-response functions and mechanistic rate parameterization.

The central object is :class:`RateSet`. It separates drug effects on division,
death, and phenotype transitions instead of collapsing them into a single
net-growth curve. For phenotype ``i`` and transition ``i -> j`` the low-density
rate laws are

.. math::

   b_i(c) = b_{0,i}\,[1-m_{b,i}(c)],

   d_i(c) = d_{0,i}+m_{d,i}(c),

   u_{ij}(c) = u_{0,ij}\,f_{ij}(c)\,[1+m_{ij}(c)] + a_{ij}(c).

``m`` is a dimensionless effect magnitude, ``f`` a non-negative fold-change,
and ``a`` an additive induction rate. The additive term matters for
drug-induced plasticity: it lets a transition be absent without treatment
(``u0 = 0``) yet appear under it, which a purely multiplicative law cannot
express because ``0 * anything = 0``.

Stochastic simulation, moment equations, observation noise, pharmacokinetics,
and initial subpopulation fractions belong to the dynamics and inference
layers rather than here.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from umimic.dynamics.states import CellType

if TYPE_CHECKING:
    from umimic.dynamics.states import ModelTopology

logger = logging.getLogger(__name__)


class DoseResponseFunction(ABC):
    """Abstract base for concentration -> effect modulation.

    Modulator sign convention
    -------------------------
    Functions used as *modulators* in :class:`RateSet` must be **increasing**
    in concentration, because the rate laws read

        birth:      b(C) = b0 * (1 - m_b(C))     -- more drug, less division
        death:      d(C) = d0 + m_d(C)           -- more drug, more death
        transition: u(C) = u0 * f(C) * (1 + m_t(C)) + a(C)

    so ``m`` and ``a`` are the *magnitude of the drug effect*, not the
    response itself. A decreasing function such as
    :class:`FourParameterLogistic` in its usual orientation inverts the
    pharmacology: as a birth modulator it makes the untreated division rate
    zero and lets the drug *increase* growth. Use
    :meth:`FourParameterLogistic.as_effect` to reorient one.

    The fold-change ``f`` is the exception: it is a multiplier, so it may
    rise or fall with concentration and is only required to stay
    non-negative. Suppression of a transition -- a drug blocking
    resensitisation, say -- belongs in ``f`` via :class:`HillFoldChange`,
    not in a negative-valued ``m``.
    """

    @abstractmethod
    def __call__(self, concentration: float | np.ndarray) -> float | np.ndarray:
        ...

    @abstractmethod
    def param_names(self) -> list[str]:
        ...

    def is_increasing(self, probes: np.ndarray | None = None) -> bool:
        """Whether the function increases with concentration.

        Checked by probing across a wide log range rather than analytically,
        so it works for any subclass.
        """
        if probes is None:
            probes = np.concatenate([[0.0], np.logspace(-3, 3, 25)])
        values = np.array([float(self(c)) for c in probes])
        return bool(np.all(np.diff(values) >= -1e-12))


@dataclass
class EmaxHill(DoseResponseFunction):
    """Emax/Hill dose-response: m(C) = Emax * C^Hill / (EC50^Hill + C^Hill).

    Returns a value in [0, Emax] that increases with concentration.
    """

    emax: float = 1.0
    ec50: float = 1.0
    hill: float = 1.0

    def __post_init__(self) -> None:
        # Same checks HillFoldChange already applies. ec50 = 0 makes the
        # denominator collapse to c**hill, so the curve jumps to emax at every
        # positive concentration and is 0/0 at c = 0; hill <= 0 inverts or
        # flattens the response. Both produce a plausible-looking fit rather
        # than an obvious failure.
        if not np.isfinite(self.emax) or self.emax < 0.0:
            raise ValueError(f"EmaxHill emax must be non-negative, got {self.emax}.")
        if not np.isfinite(self.ec50) or self.ec50 <= 0.0:
            raise ValueError(f"EmaxHill ec50 must be positive, got {self.ec50}.")
        if not np.isfinite(self.hill) or self.hill <= 0.0:
            raise ValueError(f"EmaxHill hill must be positive, got {self.hill}.")

    def __call__(self, c: float | np.ndarray) -> float | np.ndarray:
        c = np.asarray(c, dtype=float)
        c_safe = np.maximum(c, 0.0)
        return self.emax * c_safe**self.hill / (self.ec50**self.hill + c_safe**self.hill)

    def param_names(self) -> list[str]:
        return ["emax", "ec50", "hill"]


@dataclass
class HillFoldChange(DoseResponseFunction):
    r"""Hill interpolation between low- and high-dose rate multipliers.

    .. math::

       f(C) = f_0 + (f_\infty - f_0) \frac{C^h}{EC_{50}^h + C^h}.

    Unlike :class:`EmaxHill`, this returns a *fold-change* rather than an
    additive or fractional effect. It can therefore represent both induction
    (``high > low``) and suppression (``high < low``) while keeping rates
    non-negative -- which is why suppression of a transition belongs here
    rather than in a negative-valued modulation.
    """

    low: float = 1.0
    high: float = 1.0
    ec50: float = 1.0
    hill: float = 1.0

    def __post_init__(self) -> None:
        if self.low < 0.0 or self.high < 0.0:
            raise ValueError("HillFoldChange multipliers must be non-negative.")
        if self.ec50 <= 0.0:
            raise ValueError("HillFoldChange ec50 must be positive.")
        if self.hill <= 0.0:
            raise ValueError("HillFoldChange hill must be positive.")

    def __call__(self, c: float | np.ndarray) -> float | np.ndarray:
        c = np.asarray(c, dtype=float)
        c_safe = np.maximum(c, 0.0)
        occupancy = c_safe**self.hill / (self.ec50**self.hill + c_safe**self.hill)
        return self.low + (self.high - self.low) * occupancy

    def param_names(self) -> list[str]:
        return ["low", "high", "ec50", "hill"]


@dataclass
class FourParameterLogistic(DoseResponseFunction):
    """4-parameter logistic: m(C) = bottom + (top - bottom) / (1 + (C/EC50)^hill).

    Commonly used for sigmoidal dose-response curves of an *observed response*
    (e.g. viability), which **decreases** with concentration when
    ``top > bottom``.

    .. warning::
       In that orientation this is not a valid rate modulator. RateSet
       modulators must increase with concentration -- see
       :class:`DoseResponseFunction`. Passing a decreasing 4PL as
       ``birth_modulation`` sets the untreated division rate to
       ``b0 * (1 - top)`` (zero for the default ``top=1``) and makes higher
       doses *increase* growth. Use :meth:`as_effect` to convert a fitted
       viability curve into a drug-effect modulator.
    """

    top: float = 1.0
    bottom: float = 0.0
    ec50: float = 1.0
    hill: float = 1.0

    def __call__(self, c: float | np.ndarray) -> float | np.ndarray:
        c = np.asarray(c, dtype=float)
        c_safe = np.maximum(c, 1e-30)
        return self.bottom + (self.top - self.bottom) / (
            1.0 + (c_safe / self.ec50) ** self.hill
        )

    def as_effect(self) -> FourParameterLogistic:
        """Reorient into an increasing drug-effect modulator.

        Returns the complementary curve, rising from 0 at zero concentration
        to ``top - bottom`` at saturation, which is the fractional effect
        RateSet's rate laws expect.
        """
        return FourParameterLogistic(
            top=0.0,
            bottom=self.top - self.bottom,
            ec50=self.ec50,
            hill=self.hill,
        )

    def param_names(self) -> list[str]:
        return ["top", "bottom", "ec50", "hill"]


@dataclass
class ConstantRate(DoseResponseFunction):
    """Constant (no drug modulation): m(C) = value."""

    value: float = 0.0

    def __post_init__(self) -> None:
        # Every sibling class validates; this one did not, and a negative
        # constant slips through both RateSet guards: it is non-decreasing (so
        # the monotonicity probe passes) and never exceeds 1 (so the birth-peak
        # probe passes). ConstantRate(-0.5) then *raises* the division rate by
        # 50% under drug -- the exact inversion those guards exist to prevent.
        if not np.isfinite(self.value) or self.value < 0:
            raise ValueError(
                f"ConstantRate value must be finite and non-negative, got "
                f"{self.value}. Modulators express the magnitude of a drug "
                "effect; a negative value inverts its direction."
            )

    def __call__(self, c: float | np.ndarray) -> float | np.ndarray:
        return np.full_like(np.asarray(c, dtype=float), self.value)

    def param_names(self) -> list[str]:
        return ["value"]


@dataclass(frozen=True)
class PhenotypeRateProfile:
    """State-specific division, death, and drug-response parameters.

    ``birth_modulation`` is a fractional inhibition of the baseline division
    rate; ``death_modulation`` is an additive death rate. Independent
    functions per phenotype let resistance be expressed through altered
    efficacy, potency, slope, or any combination, rather than through a single
    ambiguous resistance scalar.

    Grouping these four numbers per phenotype also replaces four parallel
    dicts on :class:`RateSet` that previously had to be kept in step by hand.
    """

    birth_base: float
    death_base: float
    birth_modulation: DoseResponseFunction | None = None
    death_modulation: DoseResponseFunction | None = None

    def __post_init__(self) -> None:
        if self.birth_base < 0.0:
            raise ValueError("birth_base must be non-negative.")
        if self.death_base < 0.0:
            raise ValueError("death_base must be non-negative.")


@dataclass(frozen=True)
class TransitionRateProfile:
    """Baseline, multiplicative, and induced components of a transition.

    The resulting rate is ``base_rate * factor(C) * (1 + legacy(C)) +
    induction(C)``.

    ``factor`` is the preferred way to scale a pre-existing transition up or
    down. ``induction`` has rate units and creates genuinely de novo
    treatment-induced transitions even when ``base_rate`` is zero -- the case
    a multiplicative law cannot reach.
    """

    base_rate: float = 0.0
    factor: DoseResponseFunction | None = None
    induction: DoseResponseFunction | None = None

    def __post_init__(self) -> None:
        if self.base_rate < 0.0:
            raise ValueError("base_rate must be non-negative.")


@dataclass
class RateSet:
    """Complete set of concentration-modulated rates for the CTMC.

    Mechanistic rate parameterization:
    - birth_rate(C) = b0 * (1 - mb(C))  -- drug reduces division
    - death_rate(C) = d0 + md(C)         -- drug increases death
    - transition_rate(C) = u0 * (1 + mt(C))  -- drug may modulate transitions

    Parameter naming convention (inference ↔ model mapping):
        Inference vector    RateSet field             Description
        ──────────────────  ────────────────────────  ──────────────────────
        b0                  birth_base                Baseline birth rate
        d0_P                death_base[CellType.P]    Baseline death (P)
        d0_Q                death_base[CellType.Q]    Baseline death (Q)
        emax_death          death_modulation Emax     Max drug death effect
        ec50_death          death_modulation EC50     Half-max concentration
        hill_death          death_modulation Hill     Hill coefficient
        u_PQ                transition_base[(P, Q)]   P→Q transition rate
        u_QP                transition_base[(Q, P)]   Q→P transition rate
        u_PR                transition_base[(P, R)]   P→R transition rate
        overdispersion      (observation model)       NegBin overdispersion
    """

    # Birth rate (inference name: b0). 0.04 /h is a division-time scale of
    # ~17 h; it is NOT the population doubling time, which also depends on
    # death and transitions (~27 h for the default P/Q topology). See
    # RateSet.doubling_time.
    birth_base: float = 0.04  # 1/hour
    birth_modulation: DoseResponseFunction | None = None

    # Per-state overrides. A resistant clone needs its own division rate and,
    # critically, must not inherit the drug's birth suppression -- that is what
    # makes it resistant. An explicit entry here (including an explicit None
    # modulation) overrides the shared value above for that state.
    birth_base_by_state: dict[CellType, float] = field(default_factory=dict)
    birth_modulation_by_state: dict[CellType, DoseResponseFunction | None] = field(
        default_factory=dict
    )

    # Death rates per cell type (inference names: d0_P, d0_Q)
    death_base: dict[CellType, float] = field(
        default_factory=lambda: {CellType.P: 0.01, CellType.Q: 0.005}
    )
    death_modulation: dict[CellType, DoseResponseFunction | None] = field(
        default_factory=dict
    )

    # Transition rates between states (inference names: u_PQ, u_QP, u_PR)
    transition_base: dict[tuple[CellType, CellType], float] = field(
        default_factory=lambda: {
            (CellType.P, CellType.Q): 0.005,
            (CellType.Q, CellType.P): 0.003,
        }
    )
    transition_modulation: dict[
        tuple[CellType, CellType], DoseResponseFunction | None
    ] = field(default_factory=dict)

    # Apoptotic clearance
    clearance_rate: float = 0.1  # rate of dead cell removal

    # Preferred transition extensions, appended after the legacy fields so
    # positional construction of the original RateSet stays compatible.
    # - factor(C) is a non-negative multiplier; it may increase or suppress a
    #   pre-existing transition.
    # - induction(C) is an additive rate and can create a drug-induced
    #   transition even when transition_base is zero.
    transition_factor: dict[
        tuple[CellType, CellType], DoseResponseFunction | None
    ] = field(default_factory=dict)
    transition_induction: dict[
        tuple[CellType, CellType], DoseResponseFunction | None
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate_modulators()

    def _validate_modulators(self) -> None:
        """Reject modulators that run backwards in concentration.

        A decreasing curve inverts the pharmacology silently: as a birth
        modulator it zeroes the untreated division rate and makes higher doses
        increase growth. This is the single easiest way to misuse the rate
        laws, so it is caught at construction rather than showing up as an
        implausible fit.

        Fold-changes (`transition_factor`) are exempt by design: they are
        multipliers, so a falling curve legitimately means the drug suppresses
        that transition. They are only checked for non-negativity.
        """
        effects: list[tuple[str, DoseResponseFunction | None]] = [
            ("birth_modulation", self.birth_modulation)
        ]
        effects += [
            (f"birth_modulation_by_state[{ct.name}]", fn)
            for ct, fn in self.birth_modulation_by_state.items()
        ]
        effects += [
            (f"death_modulation[{ct.name}]", fn)
            for ct, fn in self.death_modulation.items()
        ]
        effects += [
            (f"transition_modulation[{src.name}->{tgt.name}]", fn)
            for (src, tgt), fn in self.transition_modulation.items()
        ]
        effects += [
            (f"transition_induction[{src.name}->{tgt.name}]", fn)
            for (src, tgt), fn in self.transition_induction.items()
        ]

        # Modulators are effect *magnitudes*; a negative value flips the sign
        # of the drug's action. The monotonicity check below cannot catch this
        # on its own -- a constant or an offset curve can be non-decreasing and
        # still negative everywhere.
        magnitude_probes = np.concatenate([[0.0], np.logspace(-3, 6, 40)])
        for name, fn in effects:
            if fn is None:
                continue
            floor = float(np.min([float(fn(c)) for c in magnitude_probes]))
            if floor < -1e-12:
                raise ValueError(
                    f"{name} reaches {floor:.4g} < 0. RateSet modulators express "
                    "the magnitude of a drug effect, so a negative value "
                    "reverses its direction: a negative death modulation makes "
                    "the drug protective, and a negative birth modulation makes "
                    "it mitogenic."
                )

        for name, fn in effects:
            if fn is None or fn.is_increasing():
                continue
            hint = ""
            if isinstance(fn, FourParameterLogistic):
                hint = (
                    " A FourParameterLogistic with top > bottom decreases with "
                    "concentration; call .as_effect() to reorient it."
                )
            if name.startswith("transition_"):
                hint += (
                    " To model a drug that *suppresses* this transition, put a "
                    "HillFoldChange with high < low in transition_factor "
                    "instead of a decreasing modulation."
                )
            raise ValueError(
                f"{name} decreases with concentration. RateSet modulators "
                f"express the magnitude of the drug effect and must be "
                f"non-decreasing; see DoseResponseFunction.{hint}"
            )

        # Birth modulators are *fractional* suppressors: b0 * (1 - mb(C)).
        # Above 1 the birth rate is clamped to zero, so every larger value
        # gives an identical trajectory -- a flat ridge in the likelihood, and
        # an estimate that looks like "more than complete cytostasis" when
        # read as a fraction. Death modulation is exempt: it is an additive
        # rate increment, so values above 1 are meaningful there.
        birth_effects: list[tuple[str, DoseResponseFunction | None]] = [
            ("birth_modulation", self.birth_modulation)
        ]
        birth_effects += [
            (f"birth_modulation_by_state[{ct.name}]", fn)
            for ct, fn in self.birth_modulation_by_state.items()
        ]
        saturating = np.concatenate([[0.0], np.logspace(-3, 6, 40)])
        for name, fn in birth_effects:
            if fn is None:
                continue
            peak = float(np.max([float(fn(c)) for c in saturating]))
            if peak > 1.0 + 1e-9:
                raise ValueError(
                    f"{name} reaches {peak:.4g} > 1. Birth modulation is the "
                    "suppressed *fraction* of the division rate, so 1 is "
                    "complete cytostasis; larger values clamp the rate to zero "
                    "and leave the parameter unidentifiable. Use emax <= 1, or "
                    "express extra drug effect as death modulation."
                )

        # Fold-changes may rise or fall, but a negative multiplier would make
        # the rate negative.
        probes = np.concatenate([[0.0], np.logspace(-3, 3, 25)])
        for (src, tgt), fn in self.transition_factor.items():
            if fn is None:
                continue
            if np.any(np.array([float(fn(c)) for c in probes]) < 0.0):
                raise ValueError(
                    f"transition_factor[{src.name}->{tgt.name}] is negative at "
                    "some concentration; a fold-change must stay non-negative."
                )

    def modulated_birth_base(
        self, concentration: float, cell_type: CellType | None = None
    ) -> float:
        """Density-independent part of the birth rate for a dividing state.

        B(C) = b0 * (1 - mb(C)), clamped at zero. The density correction is
        applied separately by :meth:`birth_rate` so that callers needing the
        analytic derivative with respect to population (the moment-equation
        Jacobian) can obtain B(C) without re-deriving the modulation.

        Per-state overrides take precedence, which is how a resistant clone
        divides at its own rate and escapes the drug's birth suppression.
        """
        b = self.birth_base
        modulation = self.birth_modulation
        if cell_type is not None:
            if cell_type in self.birth_base_by_state:
                b = self.birth_base_by_state[cell_type]
            if cell_type in self.birth_modulation_by_state:
                modulation = self.birth_modulation_by_state[cell_type]

        if modulation is not None:
            b *= 1.0 - float(modulation(concentration))
        return max(0.0, b)

    def birth_rate(
        self,
        concentration: float,
        total_cells: float = 0.0,
        carrying_capacity: float | None = None,
        cell_type: CellType | None = None,
    ) -> float:
        """Effective birth rate at given concentration.

        b(C) = b0 * (1 - mb(C)) * density_correction

        Args:
            concentration: Drug concentration.
            total_cells: Population entering the density term.
            carrying_capacity: K, or None for density-independent growth.
            cell_type: Dividing state, used to resolve per-state overrides.
        """
        b = self.modulated_birth_base(concentration, cell_type)
        if carrying_capacity is not None and carrying_capacity > 0:
            b *= max(0.0, 1.0 - total_cells / carrying_capacity)
        b = max(0.0, b)
        if b == 0.0 and self.birth_base > 0:
            logger.debug("Birth rate suppressed to 0 at conc=%s", concentration)
        return b

    def death_rate(self, cell_type: CellType, concentration: float) -> float:
        """Effective death rate for a cell type at given concentration.

        d(C) = d0 + md(C)
        """
        d = self.death_base.get(cell_type, 0.0)
        mod = self.death_modulation.get(cell_type)
        if mod is not None:
            d += float(mod(concentration))
        return max(0.0, d)

    def transition_rate(
        self, source: CellType, target: CellType, concentration: float
    ) -> float:
        """Effective transition rate from source to target at given concentration.

        ``u(C) = u0 * f(C) * (1 + mt(C)) + a(C)``

        ``mt`` is retained for backward compatibility. New models should use
        ``f`` (`transition_factor`) for multiplicative induction or
        suppression and ``a`` (`transition_induction`) for de novo
        drug-induced transitions, which a multiplicative term cannot express
        when ``u0 = 0``.
        """
        key = (source, target)
        u = self.transition_base.get(key, 0.0)

        factor = self.transition_factor.get(key)
        if factor is not None:
            u *= max(0.0, float(factor(concentration)))

        mod = self.transition_modulation.get(key)
        if mod is not None:
            u *= max(0.0, 1.0 + float(mod(concentration)))

        induction = self.transition_induction.get(key)
        if induction is not None:
            u += float(induction(concentration))

        return max(0.0, u)

    def net_growth_rate(self, concentration: float) -> float:
        """Single-compartment net rate b(C) - d_P(C).

        .. warning::
           This is **not** the growth rate of a multi-state model. It ignores
           transitions out of P and death in every other state, so for the
           default P/Q topology it returns 0.030 /h where the true asymptotic
           rate is 0.0254 /h (doubling times of 23 h versus 27 h). Use
           :meth:`asymptotic_growth_rate` for doubling times, GR metrics, or
           any comparison against measured growth.
        """
        b = self.birth_rate(concentration, cell_type=CellType.P)
        d = self.death_rate(CellType.P, concentration)
        return b - d

    def low_density_rate_matrix(
        self,
        concentration: float,
        topology: ModelTopology,
    ) -> np.ndarray:
        """Viable-state mean rate matrix at low density.

        Rows and columns follow ``topology.active_states`` with the apoptotic
        state removed. Birth and death enter the diagonal; a conversion
        ``source -> target`` leaves the source diagonal and enters the target
        row of the source column.

        This is the deterministic first-moment generator of the viable
        multitype process, and the object underlying asymptotic growth, stable
        phenotype fractions, and GR-style summaries.
        """
        viable = [ct for ct in topology.active_states if ct != CellType.A]
        index = {ct: i for i, ct in enumerate(viable)}
        matrix = np.zeros((len(viable), len(viable)))

        for ct in viable:
            i = index[ct]
            if ct in topology.division_states:
                # Low-density limit: the crowding factor tends to 1.
                matrix[i, i] += self.birth_rate(concentration, cell_type=ct)
            if ct in topology.death_states:
                matrix[i, i] -= self.death_rate(ct, concentration)

        for src, tgt in topology.transitions:
            if src not in index:
                continue
            rate = self.transition_rate(src, tgt, concentration)
            matrix[index[src], index[src]] -= rate
            if tgt in index:
                matrix[index[tgt], index[src]] += rate

        return matrix

    def asymptotic_growth_rate(
        self,
        concentration: float,
        topology: ModelTopology,
    ) -> float:
        """Asymptotic exponential growth rate of the viable population.

        This is the dominant eigenvalue of the low-density rate matrix, which
        is the rate the population actually approaches once the state
        distribution has relaxed. Transitions and death in non-P states both
        enter it.

        The apoptotic compartment is excluded: it is a sink fed by the viable
        states and does not influence their growth.

        Args:
            concentration: Drug concentration.
            topology: Model topology defining states and transitions.

        Returns:
            Dominant eigenvalue (1/hour). Negative means net decline.
        """
        matrix = self.low_density_rate_matrix(concentration, topology)
        if matrix.size == 0:
            return 0.0
        return float(np.max(np.linalg.eigvals(matrix).real))

    def gr_value(
        self,
        concentration: float,
        topology: ModelTopology,
        reference_concentration: float = 0.0,
    ) -> float:
        """Hafner-style normalized growth-rate inhibition value.

        Under exponential growth the GR metric reduces to

            GR(C) = 2 ** (g(C) / g(reference)) - 1

        with ``g`` the dominant low-density growth rate. GR = 1 means no
        effect, 0 complete cytostasis, and negative values net population
        loss. Unlike a viability ratio, GR is insensitive to the number of
        divisions elapsed during the assay.

        .. warning::
           This is an **asymptotic** summary. With a fitter resistant state
           the dominant eigenvalue is eventually governed by R even when R is
           negligible over a finite assay, so this can report a drug as
           ineffective that clears the sensitive population within the
           experiment. For finite-horizon comparisons use the simulated
           trajectory over the actual assay window, not this diagnostic.
        """
        reference_rate = self.asymptotic_growth_rate(
            reference_concentration, topology
        )
        if reference_rate <= 0.0:
            raise ValueError(
                "GR normalization requires a positive reference growth rate; "
                f"got {reference_rate:.6g}."
            )
        treated_rate = self.asymptotic_growth_rate(concentration, topology)
        return float(np.exp2(treated_rate / reference_rate) - 1.0)

    def finite_horizon_gr(
        self,
        concentration: float,
        topology: ModelTopology,
        initial_state: np.ndarray,
        duration: float,
        reference_concentration: float = 0.0,
    ) -> float:
        """GR value measured over a finite assay window.

        Hafner's GR metric is defined from the growth actually observed during
        an experiment:

            GR = 2 ** (log2(x_treated / x_0) / log2(x_control / x_0)) - 1

        with counts taken at the end of the assay. This is what a 72-hour
        screen measures, and it is the quantity to compare against published
        GR values.

        Prefer this to :meth:`gr_value` whenever the model contains a state
        that is negligible at the start but dominant eventually. The
        asymptotic metric is governed by the dominant eigenvalue, so a rare
        resistant clone sets it at every dose and the drug is reported as
        ineffective even when it clears the sensitive population well inside
        the window. Over a finite horizon the two answers can differ by more
        than the effect being measured.

        Args:
            concentration: Treated drug concentration.
            topology: Model topology.
            initial_state: Starting population vector.
            duration: Assay length, in the model's time units.
            reference_concentration: Untreated control concentration.

        Returns:
            GR value: 1 no effect, 0 complete cytostasis, negative net kill.
        """
        # Deferred import: ode_system depends on this module.
        from umimic.dynamics.ode_system import CellDynamicsODE

        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}.")

        initial_state = np.asarray(initial_state, dtype=float)
        # Count every non-apoptotic state. This is the viable population, which
        # is distinct from the crowding total: corpses may be excluded from
        # density while still being present in the well.
        viable = np.array(
            [0.0 if ct == CellType.A else 1.0 for ct in topology.active_states]
        )

        start = float(viable @ np.maximum(initial_state, 0.0))
        if start <= 0:
            raise ValueError("initial_state has no viable cells.")

        def endpoint(conc: float) -> float:
            ode = CellDynamicsODE(self, topology, lambda _t, _c=conc: _c)
            result = ode.solve(
                initial_state, (0.0, duration), np.array([0.0, duration])
            )
            final = np.array(
                [result.populations[ct.name][-1] for ct in topology.active_states]
            )
            return float(viable @ np.maximum(final, 0.0))

        control = endpoint(reference_concentration)
        control_growth = np.log2(max(control, 1e-300) / start)
        if control_growth <= 0:
            raise ValueError(
                "The untreated control does not grow over this window "
                f"(log2 fold change {control_growth:.3g}); GR is undefined. "
                "Lengthen the assay or check the baseline rates."
            )

        treated = endpoint(concentration)
        treated_growth = np.log2(max(treated, 1e-300) / start)
        return float(2.0 ** (treated_growth / control_growth) - 1.0)

    def doubling_time(
        self, concentration: float, topology: ModelTopology
    ) -> float:
        """Net population doubling time (hours), or inf if not growing."""
        rate = self.asymptotic_growth_rate(concentration, topology)
        if rate <= 0:
            return float("inf")
        return float(np.log(2) / rate)

    def stable_state_fractions(
        self,
        concentration: float,
        topology: ModelTopology,
    ) -> np.ndarray:
        """Relaxed distribution of cells across states, as fractions.

        This is the normalized dominant right eigenvector of the low-density
        rate matrix: the state mix an exponentially growing culture settles
        into. It is a better initial condition than "every cell is
        proliferating" for a culture that has been passaged rather than
        freshly sorted, which otherwise biases early growth and the estimated
        transition rates.

        Returns:
            Fractions over `topology.active_states`, summing to 1. The
            apoptotic compartment is assigned zero.
        """
        viable = [ct for ct in topology.active_states if ct != CellType.A]
        if not viable:
            raise ValueError("Topology has no viable states.")

        index = {ct: i for i, ct in enumerate(viable)}
        n = len(viable)
        values, vectors = np.linalg.eig(
            self.low_density_rate_matrix(concentration, topology)
        )
        dominant = vectors[:, int(np.argmax(values.real))].real
        dominant = np.abs(dominant)
        total = dominant.sum()
        if total <= 0:
            # Degenerate rate matrix; fall back to all cells proliferating.
            dominant = np.zeros(n)
            dominant[index.get(CellType.P, 0)] = 1.0
            total = 1.0

        fractions = np.zeros(topology.n_states)
        for ct, i in index.items():
            fractions[topology.state_index(ct)] = dominant[i] / total
        return fractions

    def all_rates_at(
        self,
        concentration: float,
        state: np.ndarray,
        topology: ModelTopology,
    ) -> dict[str, float]:
        """Compute all rates given concentration and current state.

        Returns a dict with named rates for inspection. The density term uses
        the topology's crowding total, so this diagnostic agrees with the
        solvers; summing every state (apoptotic included) reported birth = 0
        after a cytotoxic pulse even with the viable population far below K.

        Division is reported per dividing state as ``birth_<STATE>``. A single
        shared ``birth`` key omitted ``cell_type``, so it always returned the
        global ``birth_base`` and silently ignored ``birth_base_by_state`` and
        ``birth_modulation_by_state``: a resistant clone dividing at 0.02 was
        reported at 0.04. This is the function users reach for to check a
        resistance model, so it has to be able to represent one.
        """
        total = topology.density_total(state)
        K = topology.carrying_capacity if topology.density_dependent else None
        rates = {}
        for ct in topology.division_states:
            rates[f"birth_{ct.name}"] = self.birth_rate(
                concentration, total, K, cell_type=ct
            )
        for ct in topology.death_states:
            rates[f"death_{ct.name}"] = self.death_rate(ct, concentration)
        for src, tgt in topology.transitions:
            rates[f"trans_{src.name}_{tgt.name}"] = self.transition_rate(
                src, tgt, concentration
            )
        rates["clearance"] = self.clearance_rate
        return rates

    @classmethod
    def from_profiles(
        cls,
        phenotypes: dict[CellType, PhenotypeRateProfile],
        transitions: dict[
            tuple[CellType, CellType], TransitionRateProfile
        ] | None = None,
        *,
        clearance_rate: float = 0.1,
        topology: ModelTopology | None = None,
    ) -> RateSet:
        """Construct a RateSet from explicit per-phenotype profiles.

        The preferred general constructor for resistance models: every
        phenotype's baseline fitness and cytostatic/cytotoxic response is
        explicit, and each transition gets an independently interpretable
        baseline, fold-change, and induced component.

        Args:
            phenotypes: Profiles keyed by cell state. P is used as the shared
                default when present, otherwise the first entry.
            transitions: Optional transition profiles keyed by
                ``(source, target)``. Only edges also present in the topology
                affect simulated dynamics.
            clearance_rate: Apoptotic-cell clearance rate.
            topology: When given, every dividing state must have a profile.
                Without this check a state with no profile silently inherits
                P's drug response -- so a resistant compartment would quietly
                acquire the sensitivity it is supposed to lack.
        """
        if not phenotypes:
            raise ValueError("At least one phenotype profile is required.")
        if clearance_rate < 0.0:
            raise ValueError("clearance_rate must be non-negative.")

        if topology is not None:
            missing = [
                ct.name
                for ct in topology.division_states
                if ct not in phenotypes
            ]
            if missing:
                raise ValueError(
                    f"No PhenotypeRateProfile supplied for dividing state(s) "
                    f"{missing}. They would otherwise inherit the reference "
                    "phenotype's division rate and drug response, which is "
                    "how a resistant state silently becomes drug-sensitive."
                )

        reference_state = (
            CellType.P if CellType.P in phenotypes else next(iter(phenotypes))
        )
        reference = phenotypes[reference_state]

        birth_base_by_state: dict[CellType, float] = {}
        birth_modulation_by_state: dict[CellType, DoseResponseFunction | None] = {}
        for cell_type, profile in phenotypes.items():
            if cell_type == reference_state:
                continue
            birth_base_by_state[cell_type] = profile.birth_base
            # Keep an explicit None so the state does not inherit P's response.
            birth_modulation_by_state[cell_type] = profile.birth_modulation

        death_base = {ct: p.death_base for ct, p in phenotypes.items()}
        death_modulation = {
            ct: p.death_modulation
            for ct, p in phenotypes.items()
            if p.death_modulation is not None
        }

        transition_base: dict[tuple[CellType, CellType], float] = {}
        transition_factor: dict[
            tuple[CellType, CellType], DoseResponseFunction | None
        ] = {}
        transition_induction: dict[
            tuple[CellType, CellType], DoseResponseFunction | None
        ] = {}
        for key, profile in (transitions or {}).items():
            source, target = key
            if source == target:
                raise ValueError("A transition source and target must differ.")
            transition_base[key] = profile.base_rate
            if profile.factor is not None:
                transition_factor[key] = profile.factor
            if profile.induction is not None:
                transition_induction[key] = profile.induction

        return cls(
            birth_base=reference.birth_base,
            birth_modulation=reference.birth_modulation,
            birth_base_by_state=birth_base_by_state,
            birth_modulation_by_state=birth_modulation_by_state,
            death_base=death_base,
            death_modulation=death_modulation,
            transition_base=transition_base,
            transition_factor=transition_factor,
            transition_induction=transition_induction,
            clearance_rate=clearance_rate,
        )

    @classmethod
    def persister_resistance(
        cls,
        *,
        sensitive: PhenotypeRateProfile | None = None,
        persister: PhenotypeRateProfile | None = None,
        resistant: PhenotypeRateProfile | None = None,
        transitions: dict[
            tuple[CellType, CellType], TransitionRateProfile
        ] | None = None,
        epigenetic_reversion: float = 0.0,
        clearance_rate: float = 0.1,
        topology: ModelTopology | None = None,
    ) -> RateSet:
        """Illustrative P -> Q -> R persistence-to-resistance model.

        Three phenotypes with distinct baseline fitness and independent drug
        effects on division and death:

        - **P**: cycling, drug-sensitive.
        - **Q**: slow-cycling, reversibly drug-tolerant persisters.
        - **R**: cycling cells with stable high-level resistance and a small
          untreated fitness cost.

        The default transitions encode treatment-induced entry into Q,
        suppression of Q -> P resensitisation during treatment, and slower
        treatment-induced stabilisation of persisters into R -- the
        persister-to-resistant route that the additive `induction` term exists
        to represent, since it is absent without drug.

        Use :meth:`ModelTopology.persister_resistance` for a topology carrying
        the matching edges. These constants are illustrative, not fitted;
        replace them with inferred or experimentally constrained profiles.

        Args:
            epigenetic_reversion: Rate of R -> Q reversion. **Defaults to 0.**
                A per-cell reversion rate is only meaningful when R is an
                epigenetically stable state. If R is a mutant clone, cells do
                not revert -- apparent resensitisation during a drug holiday
                is competitive dilution driven by R's fitness cost, which the
                birth and death rates already capture. Setting this for a
                genetic R double-counts that effect.
            topology: Forwarded to :meth:`from_profiles` so that every
                dividing state is checked for an explicit profile.
        """
        if epigenetic_reversion < 0.0:
            raise ValueError("epigenetic_reversion must be non-negative.")

        if sensitive is None:
            sensitive = PhenotypeRateProfile(
                birth_base=0.04,
                death_base=0.01,
                birth_modulation=EmaxHill(emax=0.90, ec50=1.0, hill=1.5),
                death_modulation=EmaxHill(emax=0.06, ec50=1.0, hill=1.5),
            )
        if persister is None:
            persister = PhenotypeRateProfile(
                birth_base=0.002,
                death_base=0.003,
                birth_modulation=EmaxHill(emax=0.20, ec50=3.0, hill=1.2),
                death_modulation=EmaxHill(emax=0.004, ec50=3.0, hill=1.2),
            )
        if resistant is None:
            # Resistant at therapeutic exposure but not invincible: a much
            # higher EC50 rather than a flat absence of response, so the model
            # still shows dose-dependence at high exposure instead of
            # reporting a drug that never works at any dose.
            resistant = PhenotypeRateProfile(
                birth_base=0.036,
                death_base=0.010,
                birth_modulation=EmaxHill(emax=0.80, ec50=30.0, hill=1.5),
                death_modulation=EmaxHill(emax=0.030, ec50=30.0, hill=1.5),
            )

        if transitions is None:
            transitions = {
                # Treatment-induced entry into drug-tolerant persistence.
                (CellType.P, CellType.Q): TransitionRateProfile(
                    base_rate=1e-4,
                    induction=EmaxHill(emax=8e-3, ec50=0.5, hill=2.0),
                ),
                # Drug blocks resensitisation: a fold-change, not a negative
                # modulation.
                (CellType.Q, CellType.P): TransitionRateProfile(
                    base_rate=3e-3,
                    factor=HillFoldChange(low=1.0, high=0.1, ec50=0.5, hill=2.0),
                ),
                # Persister -> stable resistance. Absent without drug, which
                # only the additive induction term can express.
                (CellType.Q, CellType.R): TransitionRateProfile(
                    base_rate=0.0,
                    induction=EmaxHill(emax=2e-4, ec50=1.0, hill=2.0),
                ),
                # Reversion, off unless the caller declares R epigenetic.
                (CellType.R, CellType.Q): TransitionRateProfile(
                    base_rate=epigenetic_reversion,
                    factor=HillFoldChange(low=1.0, high=0.2, ec50=1.0, hill=2.0),
                ),
            }

        return cls.from_profiles(
            {
                CellType.P: sensitive,
                CellType.Q: persister,
                CellType.R: resistant,
            },
            transitions,
            clearance_rate=clearance_rate,
            topology=topology,
        )

    #: Backwards-compatible alias.
    adaptive_resistance_model = persister_resistance

    @classmethod
    def cytotoxic_drug(
        cls,
        b0: float = 0.04,
        d0: float = 0.01,
        emax_death: float = 0.05,
        ec50_death: float = 1.0,
        hill_death: float = 1.5,
        quiescent_sensitivity: float = 0.0,
    ) -> RateSet:
        """Create a RateSet for a purely cytotoxic drug (increases death only).

        Args:
            quiescent_sensitivity: Q's drug-induced death relative to P's, in
                [0, 1]. **This is an explicit modelling choice, not a
                neutral default.** 0 treats quiescent cells as fully
                refractory, which matches cell-cycle-specific agents
                (antimetabolites, taxanes) but understates kill for agents
                active against non-cycling cells (many alkylators, radiation).
                In a Q-rich population the two extremes give very different
                total kill, so set it deliberately.
        """
        return cls(
            birth_base=b0,
            birth_modulation=None,
            death_base={CellType.P: d0, CellType.Q: d0 * 0.5},
            death_modulation=cls._death_modulation(
                emax_death, ec50_death, hill_death, quiescent_sensitivity
            ),
        )

    @staticmethod
    def _death_modulation(
        emax: float,
        ec50: float,
        hill: float,
        quiescent_sensitivity: float,
    ) -> dict[CellType, DoseResponseFunction]:
        """Death modulation for P, and for Q scaled by its sensitivity."""
        if not 0.0 <= quiescent_sensitivity <= 1.0:
            raise ValueError(
                "quiescent_sensitivity must lie in [0, 1], got "
                f"{quiescent_sensitivity}."
            )
        modulation = {CellType.P: EmaxHill(emax=emax, ec50=ec50, hill=hill)}
        if quiescent_sensitivity > 0:
            modulation[CellType.Q] = EmaxHill(
                emax=emax * quiescent_sensitivity, ec50=ec50, hill=hill
            )
        return modulation

    @classmethod
    def cytostatic_drug(
        cls,
        b0: float = 0.04,
        d0: float = 0.01,
        emax_birth: float = 0.8,
        ec50_birth: float = 1.0,
        hill_birth: float = 1.5,
    ) -> RateSet:
        """Create a RateSet for a purely cytostatic drug (reduces birth only)."""
        return cls(
            birth_base=b0,
            birth_modulation=EmaxHill(
                emax=emax_birth, ec50=ec50_birth, hill=hill_birth
            ),
            death_base={CellType.P: d0, CellType.Q: d0 * 0.5},
        )

    @classmethod
    def resistant_clone(
        cls,
        b0: float = 0.04,
        d0: float = 0.01,
        emax_death: float = 0.05,
        ec50_death: float = 1.0,
        hill_death: float = 1.5,
        resistance: float = 1.0,
        fitness_cost: float = 0.0,
        u_PR: float = 1e-6,
        u_PQ: float = 0.005,
        u_QP: float = 0.003,
    ) -> RateSet:
        """RateSet for a sensitive P/Q population plus a resistant R clone.

        R proliferates at ``b0 * (1 - fitness_cost)`` and its drug sensitivity
        is scaled by ``1 - resistance``: at ``resistance=1`` the clone is fully
        insensitive and grows under treatment while P is killed, which is the
        behaviour a resistance model has to reproduce.

        The P->R edge is parameterised here rather than left to the class
        default. ``ModelTopology.four_state`` declares that edge, but the
        default ``transition_base`` covers only P<->Q, so without ``u_PR`` the
        transition rate is exactly zero and resistance can never arise de novo
        -- the clone could only ever appear by seeding R in the initial state.

        Args:
            b0: Baseline division rate for P.
            d0: Baseline death rate for P.
            emax_death, ec50_death, hill_death: Drug effect on the sensitive
                population.
            resistance: 0 = as sensitive as P, 1 = fully resistant.
            fitness_cost: Fractional reduction of R's division rate relative
                to P, i.e. the cost of carrying resistance.
            u_PR: P->R acquisition rate per cell per hour. The default is a
                per-division mutation-scale rate; set it to 0 for a
                pre-existing-resistance-only model.
            u_PQ, u_QP: Quiescence entry/exit rates.
        """
        if not 0.0 <= resistance <= 1.0:
            raise ValueError(f"resistance must lie in [0, 1], got {resistance}.")
        if not 0.0 <= fitness_cost < 1.0:
            raise ValueError(f"fitness_cost must lie in [0, 1), got {fitness_cost}.")
        if u_PR < 0:
            raise ValueError(f"u_PR must be non-negative, got {u_PR}.")

        death_modulation = {
            CellType.P: EmaxHill(emax=emax_death, ec50=ec50_death, hill=hill_death),
        }
        residual = emax_death * (1.0 - resistance)
        if residual > 0:
            death_modulation[CellType.R] = EmaxHill(
                emax=residual, ec50=ec50_death, hill=hill_death
            )

        return cls(
            birth_base=b0,
            birth_base_by_state={CellType.R: b0 * (1.0 - fitness_cost)},
            # Explicit None: R does not inherit any cytostatic birth
            # suppression applied to the sensitive population.
            birth_modulation_by_state={CellType.R: None},
            death_base={CellType.P: d0, CellType.Q: d0 * 0.5, CellType.R: d0},
            death_modulation=death_modulation,
            transition_base={
                (CellType.P, CellType.Q): u_PQ,
                (CellType.Q, CellType.P): u_QP,
                (CellType.P, CellType.R): u_PR,
            },
        )

    @classmethod
    def mixed_drug(
        cls,
        b0: float = 0.04,
        d0: float = 0.01,
        emax_death: float = 0.05,
        ec50_death: float = 1.0,
        hill_death: float = 1.5,
        emax_birth: float = 0.8,
        ec50_birth: float = 1.0,
        hill_birth: float = 1.5,
        quiescent_sensitivity: float = 0.0,
    ) -> RateSet:
        """Create a RateSet for a mixed drug: raises death and reduces birth.

        The two effects have independent Emax/EC50/Hill parameters, so a mixed
        agent is not simply the average of the two pure mechanisms. Keeping
        them separate is what would let the variance signature distinguish
        cytotoxic from cytostatic action -- in principle. That separation is
        underpowered at realistic counting noise; see
        umimic/inference/SCIENTIFIC_ASSUMPTIONS.md section 2a before relying
        on a fitted split.

        See :meth:`cytotoxic_drug` for the meaning of `quiescent_sensitivity`;
        the default of 0 leaves quiescent cells refractory to the cytotoxic
        component.
        """
        return cls(
            birth_base=b0,
            birth_modulation=EmaxHill(
                emax=emax_birth, ec50=ec50_birth, hill=hill_birth
            ),
            death_base={CellType.P: d0, CellType.Q: d0 * 0.5},
            death_modulation=cls._death_modulation(
                emax_death, ec50_death, hill_death, quiescent_sensitivity
            ),
        )
