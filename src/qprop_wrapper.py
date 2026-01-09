from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Union

Number = Union[int, float]

@dataclass(frozen=True)
class QpropAirfoilParams:
    CL0: Number
    CL_a: Number
    CLmin: Number
    CLmax: Number
    CD0: Number
    CD2u: Number
    CD2l: Number
    CLCD0: Number
    REref: Number
    REexp: Number

    @staticmethod
    def from_fit_dict(d: Dict[str, Number]) -> "QpropAirfoilParams":
        # Adjust these keys ONLY if your fitter uses different names.
        return QpropAirfoilParams(
            CL0=d["CL0"],
            CL_a=d["CL_a"],
            CLmin=d["Cl_min"],
            CLmax=d["Cl_max"],
            CD0=d["CD0"],
            CD2u=d["Cd2u"],
            CD2l=d["Cd2l"],
            CLCD0=d["CLCD0"],
            REref=d["Re_ref"],
            REexp=d["Re_exp"],
        )

@dataclass(frozen=True)
class QpropStation:
    r_m: Number
    chord_m: Number
    beta_deg: Number
    af: QpropAirfoilParams

def write_qprop_propfile_inline_af(
    path: Path,
    *,
    Nblades: int,
    R_m: Number,
    default_af: QpropAirfoilParams,
    stations: Sequence[QpropStation],
    # Keep these at identity since you’re writing absolute units:
    Rfac: Number = 1.0,
    Cfac: Number = 1.0,
    Bfac: Number = 1.0,
    Radd: Number = 0.0,
    Cadd: Number = 0.0,
    Badd: Number = 0.0,
) -> None:
    if not stations:
        raise ValueError("Need at least one station")

    # Ensure increasing radii and within tip
    r_last = -1.0
    for st in stations:
        if st.r_m <= r_last:
            raise ValueError("Stations must have strictly increasing r")
        if st.r_m >= R_m:
            raise ValueError("Station radii should be < R (stop at e.g. 0.98R)")
        r_last = st.r_m

    lines: List[str] = []
    lines.append(f"{Nblades:d}  {R_m:.10g}  ! Nblades  [ R ]\n\n")

    # Defaults (lines 3–6)
    lines.append(f"{default_af.CL0:.10g}  {default_af.CL_a:.10g}  ! CL0     CL_a\n")
    lines.append(f"{default_af.CLmin:.10g}  {default_af.CLmax:.10g}  ! CLmin   CLmax\n\n")

    lines.append(
        f"{default_af.CD0:.10g}  {default_af.CD2u:.10g}  {default_af.CD2l:.10g}  {default_af.CLCD0:.10g}"
        f"  !  CD0    CD2u   CD2l   CLCD0\n"
    )
    lines.append(f"{default_af.REref:.10g}  {default_af.REexp:.10g}              !  REref  REexp\n\n")

    # Identity scaling since you’re writing absolute units
    lines.append(f"{Rfac:.10g}  {Cfac:.10g}  {Bfac:.10g}  !  Rfac   Cfac   Bfac\n")
    lines.append(f"{Radd:.10g}  {Cadd:.10g}  {Badd:.10g}  !  Radd   Cadd   Badd\n\n")

    # Station header
    lines.append("#  r  chord  beta [  CL0  CL_a   CLmin CLmax  CD0   CD2u   CD2l   CLCD0  REref  REexp ]\n")

    # Full override on every station line (your preference)
    for st in stations:
        af = st.af
        lines.append(
            f"{st.r_m:.10g}  {st.chord_m:.10g}  {st.beta_deg:.10g}  "
            f"{af.CL0:.10g}  {af.CL_a:.10g}  {af.CLmin:.10g}  {af.CLmax:.10g}  "
            f"{af.CD0:.10g}  {af.CD2u:.10g}  {af.CD2l:.10g}  {af.CLCD0:.10g}  "
            f"{af.REref:.10g}  {af.REexp:.10g}\n"
        )

    path.write_text("".join(lines), encoding="utf-8")
