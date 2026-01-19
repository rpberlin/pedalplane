from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from scipy.optimize import minimize 
from pathlib import Path
import numpy as np
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union, Any

QPROP_EXE = "/Users/ryanblanchard/myApplications/Qprop/bin/qprop"

Number = Union[int, float]
_FLOAT = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"


# -------------------------
# Airfoil parameter mapping
# -------------------------

@dataclass(frozen=True)
class QpropAirfoilParams:
    # In the exact order QPROP expects on station lines:
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
    def from_fit_dict(d: Mapping[str, Number]) -> "QpropAirfoilParams":
        """
        Exact mapping from your fitter output, e.g.
        {'CL0','CL_a','CD0','Cd2u','Cd2l','CLCD0','Re_ref','Re_exp','Cl_min','Cl_max'}
        """
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


# -------------------------
# Propeller geometry classes
# -------------------------

@dataclass
class PropSection:
    """
    One prop blade section (absolute units).
      r_m: radius from axis [m]
      chord_m: chord length [m]
      beta_deg: geometric pitch angle [deg]
      airfoil_name: optional label (for debugging / provenance)
      af_fit: your fitted QPROP parameter dict for this section's airfoil
    """
    r_m: float
    chord_m: float
    beta_deg: float
    af_fit: Dict[str, Number]
    airfoil_name: str = ""  # optional, not used by QPROP in inline format


@dataclass
class Propeller:
    diameter_m: float
    hub_diameter_m: float
    chord_ref_m: float
    n_blades: int
    sections: List[PropSection] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.diameter_m <= 0:
            raise ValueError("diameter_m must be > 0")
        if self.hub_diameter_m < 0 or self.hub_diameter_m >= self.diameter_m:
            raise ValueError("hub_diameter_m must be >=0 and < diameter_m")
        if self.n_blades < 1:
            raise ValueError("n_blades must be >= 1")

    @property
    def radius_m(self) -> float:
        return 0.5 * self.diameter_m

    @property
    def hub_radius_m(self) -> float:
        return 0.5 * self.hub_diameter_m

    def add_section(
        self,
        *,
        r_m: float,
        chord_m: float,
        beta_deg: float,
        af_fit: Dict[str, Number],
        airfoil_name: str = "",
    ) -> None:
        self.sections.append(
            PropSection(
                r_m=float(r_m),
                chord_m=float(chord_m),
                beta_deg=float(beta_deg),
                af_fit=dict(af_fit),
                airfoil_name=str(airfoil_name),
            )
        )


    def getXparamsfromProp(self):
        beta_angles = []
        chord_ratios = []
        radius_ratios = []
        for section in self.sections:
            beta, chord, radius = section.beta_deg,  section.chord_m, section.r_m
            chord_ratio = chord/self.chord_ref_m
            radius_ratio = (radius-self.hub_radius_m)/(self.radius_m - self.hub_radius_m)
            beta_angles.append(beta/100)
            chord_ratios.append(chord_ratio)
            radius_ratios.append(radius_ratio)
        x = beta_angles + chord_ratios + radius_ratios[1:-1]
        return x


    def setParamsfromX(self,X):
        nParamsIn = len(X)
        n_sections = len(self.sections)
        i0_chord_ratios = n_sections
        i0_radius_ratios = 2*n_sections
        nPropParams = 3*n_sections-2 #the first and last radius ratios canot be changed
        if nParamsIn != nPropParams:
            raise ValueError("Mismatch on parameters")
        for i, section in enumerate(self.sections):
            section.beta_deg = 100*X[i]
            section.chord_m = self.chord_ref_m * X[i0_chord_ratios+i]
            if i >0 and i<n_sections-1:
                section.radius_m = self.hub_radius_m + X[i0_radius_ratios+i-1]*(self.radius_m - self.hub_radius_m)
        return
    
    def print_section_summaries(self):
        print('idx \tr/R\tR (m) \tbeta (deg) \tchord (m) \tchrd_rat\tName')
        for i, section in enumerate(self.sections):
            r_ratio = (section.r_m-self.hub_radius_m)/(self.radius_m - self.hub_radius_m)
            chord_ratio = section.chord_m/self.chord_ref_m
            print(f'{i:d} \t{r_ratio:.3f} \t{   section.r_m:.3f} \t{section.beta_deg:.2f} \t{section.chord_m:.3f}\t{chord_ratio:.3f} \t{section.airfoil_name} ')
    
    def maximize_static_thrust(self, power=300, U_axial = 10):
        x0 = self.getXparamsfromProp()
        res = minimize(
            self.neg_thrust_sim_standalone,
            x0,
            args=(power, 0.0),      # power, Uaxial
            method="nelder-mead",
            options={
                "disp": True,
                "eps": 1e-1,
                "maxiter": 200
                }
        )
        return res
    
    def calc_thrust_and_rpm_from_power(self, power=300,U_axial=100, max_iter =200):
        rpm0 = 1000
        urf = 0.8
        tol = 0.0001
        it = 0
        res0 = self.evaluate_qprop(rpm=rpm0,U_axial=U_axial,keep_dir=False)
        pow0 = res0.P_shaft_W
        err = np.abs(pow0-power)
        while err > tol and it < max_iter:
            #print(it,rpm0,pow0,err)
            rpmstar = rpm0*(power/pow0) ** (1/3)
            rpm0 = rpm0*(1-urf) + rpmstar*urf
            res0 = self.evaluate_qprop(rpm=rpm0,U_axial=U_axial,keep_dir=False)
            pow0 = res0.P_shaft_W
            err = np.abs((pow0-power)/power)
            it = it +1
        thrust = res0.T_N
        print(f'Thrust: {thrust:.4f}')
        return thrust, rpm0
            


        
   
    def neg_thrust_sim_standalone(self, X, power=100, U_axial=100):
        try:
            self.setParamsfromX(X)
            #res = self.evaluate_qprop(rpm=rpm, U_axial=Uaxial, keep_dir=False)
            thrust, rpm = self.calc_thrust_and_rpm_from_power(power=power,U_axial=U_axial)
            #print("funCalled:", thrust, rpm, X)
            return -thrust
        except Exception:
            print("Failed: ", X)
            return 1e9

    def validate(self) -> None:
        if not self.sections:
            raise ValueError("Propeller has no sections.")

        rs = [s.r_m for s in self.sections]
        if rs != sorted(rs):
            raise ValueError("Sections must be in strictly increasing r_m order.")
        if any(r <= 0 for r in rs):
            raise ValueError("All section radii must be > 0.")
        if any(r < self.hub_radius_m for r in rs):
            raise ValueError("All section radii must be > hub_radius_m (hub cutout).")
        if any(r >= self.radius_m for r in rs):
            raise ValueError("All section radii must be < tip radius (use e.g. 0.98R).")
        if any(s.chord_m <= 0 for s in self.sections):
            raise ValueError("All chord_m must be > 0.")
        if any(abs(s.beta_deg) > 89 for s in self.sections):
            raise ValueError("beta_deg magnitude looks unphysical (> 89 deg).")

        # Ensure each section has required fit keys
        required = {"CL0","CL_a","Cl_min","Cl_max","CD0","Cd2u","Cd2l","CLCD0","Re_ref","Re_exp"}
        for i, s in enumerate(self.sections):
            missing = required.difference(s.af_fit.keys())
            if missing:
                raise KeyError(f"Section {i} missing airfoil fit keys: {sorted(missing)}")

    def to_qprop_prop_text(self, default_section_index: Optional[int] = None) -> str:
        """
        Build a Drela/QPROP inline-airfoil .prop content string.
        Full airfoil overrides at every station line (incl. REref/REexp).
        """
        self.validate()

        if default_section_index is None:
            default_section_index = len(self.sections) // 2
        if not (0 <= default_section_index < len(self.sections)):
            raise ValueError("default_section_index out of range")

        default_af = QpropAirfoilParams.from_fit_dict(self.sections[default_section_index].af_fit)

        R = self.radius_m

        lines: List[str] = []
        lines.append('placeholdername\n')
        lines.append(f"{self.n_blades:d} {self.radius_m:.10g} {self.hub_radius_m:.10g} ! Nblades  PropR hubR\n\n")

        # default airfoil params (required, even if overridden everywhere)
        lines.append(f"{default_af.CL0:.10g}  {default_af.CL_a:.10g}  ! CL0     CL_a\n")
        lines.append(f"{default_af.CLmin:.10g}  {default_af.CLmax:.10g}  ! CLmin   CLmax\n\n")

        lines.append(
            f"{default_af.CD0:.10g}  {default_af.CD2u:.10g}  {default_af.CLCD0:.10g}"
            f"  !  CD0    CD2u    CLCD0\n"
        )
        lines.append(f"{default_af.REref:.10g}  {default_af.REexp:.10g}              !  REref  REexp\n\n")

        # identity scaling/offsets since we're writing absolute units
        lines.append("1.0  1.0  1.0  !  Rfac   Cfac   Bfac\n")
        lines.append("0.0  0.0  0.0  !  Radd   Cadd   Badd\n\n")

        lines.append("#  r  chord  beta [  CL0  CL_a   CLmin CLmax  CD0   CD2u   CD2l   CLCD0  REref  REexp ]\n")

        for sec in self.sections:
            af = QpropAirfoilParams.from_fit_dict(sec.af_fit)
            lines.append(
                f"{sec.r_m:.10g}  {sec.chord_m:.10g}  {sec.beta_deg:.10g}  "
                f"{af.CL0:.10g}  {af.CL_a:.10g}  {af.CLmin:.10g}  {af.CLmax:.10g}  "
                f"{af.CD0:.10g}  {af.CD2u:.10g}  {af.CD2l:.10g}  {af.CLCD0:.10g}  "
                f"{af.REref:.10g}  {af.REexp:.10g}\n"
            )

        return "".join(lines)

    def evaluate_qprop(
        self,
        *,
        rpm: float,
        U_axial: float,
        default_section_index: Optional[int] = None,
        keep_dir: bool = False,
        timeout_s: int = 60,
    ) -> "QpropPointResult":
        prop_text = self.to_qprop_prop_text(default_section_index=default_section_index)
        return qprop_evaluate_point(
            prop_text=prop_text,
            rpm=float(rpm),
            U_axial=float(U_axial),
            keep_dir=keep_dir,
            timeout_s=timeout_s,
        )


# -------------------------
# QPROP execution + parsing
# -------------------------

@dataclass(frozen=True)
class QpropPointResult:
    U_axial: float
    rpm: float
    T_N: float
    Q_Nm: float
    P_shaft_W: float
    summary: Dict[str, float]
    radial: List[Dict[str, float]]
    stdout: str = ""


def _write_motor_type0(path: Path) -> None:
    path.write_text("dummy\n\n1\n0.001 !Rmotor \n 0.001 !Io \n1.0 !Kv (rpm/volt)\n", encoding="utf-8")


def _run_qprop_cmd(
    qprop_exe: str,
    cwd: Path,
    vel: float,
    rpm: float,
    prop_filename: str = "gen.prop",
    mot_filename: str = "dummy.mot",
    timeout_s: int = 60,
) -> str:
    cmd = [qprop_exe, prop_filename, mot_filename, f"{vel:.10g}", f"{rpm:.10g}"]
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "QPROP failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Return code: {proc.returncode}\n"
            f"Output:\n{proc.stdout}"
        )
    return proc.stdout


def _parse_T_Q(stdout: str) -> Tuple[float, float]:
    lines = stdout.splitlines()

    header_idx = None
    for i, ln in enumerate(lines):
        s = ln.strip().lower()
        if ("rpm" in s) and ("t(" in s or "t " in s) and ("q(" in s or "q " in s):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not locate summary header containing rpm/T/Q in QPROP output.")

    headers = re.split(r"\s+", lines[header_idx].strip())

    def find_col(prefix: str) -> int:
        for j, h in enumerate(headers):
            if h.lower().startswith(prefix.lower()):
                return j
        return -1

    t_idx = find_col("T(")  # often T(N)
    q_idx = find_col("Q(")  # often Q(N-m)
    if t_idx < 0 or q_idx < 0:
        for j, h in enumerate(headers):
            hl = h.lower()
            if t_idx < 0 and hl.startswith("t"):
                t_idx = j
            if q_idx < 0 and hl.startswith("q"):
                q_idx = j
        if t_idx < 0 or q_idx < 0:
            raise ValueError(f"Found header but couldn't identify T/Q columns: {headers}")

    num_re = re.compile(rf"^\s*({_FLOAT})(\s+({_FLOAT}))*\s*$")
    row_vals: Optional[List[float]] = None
    for ln in lines[header_idx + 1 :]:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if num_re.match(s):
            row_vals = [float(x) for x in re.split(r"\s+", s)]
            break

    if row_vals is None:
        raise ValueError("Could not find numeric summary row after header.")
    if t_idx >= len(row_vals) or q_idx >= len(row_vals):
        raise ValueError("Summary row shorter than header; cannot extract T/Q.")

    return float(row_vals[t_idx]), float(row_vals[q_idx])


def qprop_evaluate_point(
    *,
    prop_text: str,
    rpm: float,
    U_axial: float,
    keep_dir: bool = False,
    timeout_s: int = 60,
) -> QpropPointResult:
    tmp = tempfile.mkdtemp(prefix="qprop_point_")
    workdir = Path(tmp)

    try:
        (workdir / "gen.prop").write_text(prop_text, encoding="utf-8")
        _write_motor_type0(workdir / "dummy.mot")

        stdout = _run_qprop_cmd(
            qprop_exe=QPROP_EXE,
            cwd=workdir,
            vel=U_axial,
            rpm=rpm,
            timeout_s=timeout_s,
        )

        #T, Q = _parse_T_Q(stdout)
        summary, radial = parse_qprop_single_point(stdout)
        T = summary["T_N"]
        Q = summary["Q_N_m"]  # from "Q(N-m)" -> "Q_N_m"
        omega = 2.0 * math.pi * rpm / 60.0
        P = Q * omega

        return QpropPointResult(
            U_axial=U_axial,
            rpm=rpm,
            T_N=T,
            Q_Nm=Q,
            P_shaft_W=P,
            stdout=stdout if keep_dir else "",
            summary=summary,
            radial=radial,
        )
    finally:
        if keep_dir:
            print(f"[debug] kept temp directory: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)

def parse_qprop_single_point(stdout: str) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """
    Returns:
      summary: dict of scalar values (one operating point), keys normalized from the QPROP header
      radial:  dict of lists (columnar), keys normalized from the QPROP radial header
              so you can do: plt.plot(radial['radius'], radial['Mach'])
    Requires helper functions already defined in your script:
      - _strip_leading_hash(line: str) -> str
      - _normalize_headers(tokens: List[str]) -> List[str]
      - _extract_floats_with_repair(line: str) -> List[float]
      - reformat_radial(rows: List[Dict[str, float]]) -> Dict[str, List[float]]
    """
    lines = stdout.splitlines()

    # ----------------------------
    # 1) Find summary header index
    # ----------------------------
    summary_header_idx: Optional[int] = None
    for i, ln in enumerate(lines):
        s = ln.lower()
        if "v(m/s" in s and "rpm" in s and "t(n" in s and "q(n" in s:
            summary_header_idx = i
            break
    if summary_header_idx is None:
        raise ValueError("Could not locate summary header line (V(m/s), rpm, T(N), Q(N-m)).")

    summary_header_tokens = re.split(r"\s+", _strip_leading_hash(lines[summary_header_idx]).strip())
    summary_cols = _normalize_headers([t for t in summary_header_tokens if t])
    expected_n = len(summary_cols)

    # ----------------------------
    # 2) Find radial header index (so we can stop summary scan before it)
    # ----------------------------
    radial_header_idx: Optional[int] = None
    for i in range(summary_header_idx + 1, len(lines)):
        s = lines[i].lower()
        if "radius" in s and "chord" in s and "beta" in s and "cl" in s and "cd" in s and "re" in s:
            radial_header_idx = i
            break

    # ----------------------------
    # 3) Parse summary numeric row
    #    - must be '#' prefixed
    #    - use float extractor with glue repair
    #    - stop scanning once radial header is reached
    # ----------------------------
    summary_vals: Optional[List[float]] = None

    stop_i = radial_header_idx if radial_header_idx is not None else len(lines)
    for ln in lines[summary_header_idx + 1 : stop_i]:
        s = ln.strip()
        if not s:
            continue
        # In QPROP 1.22, the summary numeric line is usually "#  0.000 1000 ..."
        if not s.startswith("#"):
            continue

        try:
            vals = _extract_floats_with_repair(s)
        except ValueError:
            continue

        if len(vals) >= expected_n:
            summary_vals = vals[:expected_n]
            break

    if summary_vals is None:
        raise ValueError(
            "Found summary header but could not parse a numeric summary row "
            "(looked only at '#' lines before the radial table)."
        )

    summary = {summary_cols[j]: float(summary_vals[j]) for j in range(expected_n)}

    # ----------------------------
    # 4) Find radial header (if not already found)
    # ----------------------------
    if radial_header_idx is None:
        # fallback: first occurrence anywhere after summary
        for i, ln in enumerate(lines):
            s = ln.lower()
            if i > summary_header_idx and "radius" in s and "chord" in s and "beta" in s and "cl" in s:
                radial_header_idx = i
                break
    if radial_header_idx is None:
        raise ValueError("Could not locate radial header line (radius/chord/beta/Cl/...).")

    radial_header_tokens = re.split(r"\s+", _strip_leading_hash(lines[radial_header_idx]).strip())
    radial_cols = _normalize_headers([t for t in radial_header_tokens if t])
    n_rad_cols = len(radial_cols)

    # ----------------------------
    # 5) Parse radial numeric rows
    #    - radial data lines are typically NOT '#' prefixed
    #    - stop at blank line or next comment block after we started
    # ----------------------------
    radial_rows: List[Dict[str, float]] = []
    num_re = re.compile(rf"^\s*({_FLOAT})(\s+({_FLOAT}))*\s*$")

    for ln in lines[radial_header_idx + 1 :]:
        s = ln.strip()
        if not s:
            if radial_rows:
                break
            continue
        if s.startswith("#"):
            if radial_rows:
                break
            continue

        # radial lines should be whitespace-delimited floats (no glue seen here typically)
        if not num_re.match(s):
            if radial_rows:
                break
            continue

        vals = [float(x) for x in re.split(r"\s+", s)]
        if len(vals) < n_rad_cols:
            # skip partial line
            continue
        vals = vals[:n_rad_cols]
        radial_rows.append({radial_cols[j]: vals[j] for j in range(n_rad_cols)})

    radial = reformat_radial(radial_rows)

    return summary, radial

def XXXparse_qprop_single_point(stdout: str) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Returns:
      summary: dict with keys like V_m_s, rpm, T_N, Q_N_m, Pshaft_W, ...
      radial: list of dict rows with keys like radius, chord, beta, Cl, Cd, Re, ...
    """
    lines = stdout.splitlines()

    # ---- 1) Find summary header ----
    # In your output it is:
    # #  V(m/s) rpm Dbeta T(N) Q(N-m) ...
    summary_header_idx: Optional[int] = None
    for i, ln in enumerate(lines):
        s = ln.lower()
        if "v(m/s" in s and "rpm" in s and "t(n" in s and "q(n" in s:
            summary_header_idx = i
            break
    if summary_header_idx is None:
        raise ValueError("Could not locate summary header line (V(m/s), rpm, T(N), Q(N-m)).")

    summary_header_tokens = re.split(r"\s+", _strip_leading_hash(lines[summary_header_idx]))
    summary_cols = _normalize_headers(summary_header_tokens)

    # ---- 2) Parse the first numeric row after the summary header ----
    num_re = re.compile(rf"^\s*#?\s*({_FLOAT})(\s+({_FLOAT}))*\s*$")
    summary_vals: Optional[List[float]] = None
    for ln in lines[summary_header_idx + 1:]:
        s = ln.strip()
        if not s:
            continue
        # allow the summary numeric line to start with '#'
        if num_re.match(s):
            parts = re.split(r"\s+", _strip_leading_hash(s))
            summary_vals = [float(x) for x in parts]
            break
        # ignore other comment lines
        continue

    if summary_vals is None:
        raise ValueError("Found summary header but could not parse a numeric summary row.")

    # sometimes there are extra tokens; trim to header length
    if len(summary_vals) > len(summary_cols):
        summary_vals = summary_vals[:len(summary_cols)]
    if len(summary_vals) != len(summary_cols):
        raise ValueError(f"Summary row length mismatch: {len(summary_vals)} vs {len(summary_cols)}")

    summary = {summary_cols[j]: summary_vals[j] for j in range(len(summary_cols))}

    # ---- 3) Find radial header ----
    # In your output it is:
    # #  radius chord beta Cl Cd Re Mach ...
    radial_header_idx: Optional[int] = None
    for i, ln in enumerate(lines):
        s = ln.lower()
        if "radius" in s and "chord" in s and "beta" in s and "cl" in s and "cd" in s and "re" in s:
            # prefer the later one (radial header comes after summary)
            if summary_header_idx is not None and i > summary_header_idx:
                radial_header_idx = i
                break
    if radial_header_idx is None:
        # Some builds print it slightly differently; still try first occurrence
        for i, ln in enumerate(lines):
            s = ln.lower()
            if "radius" in s and "chord" in s and "beta" in s and "cl" in s:
                radial_header_idx = i
                break
    if radial_header_idx is None:
        raise ValueError("Could not locate radial header line (radius/chord/beta/Cl/...).")

    radial_header_tokens = re.split(r"\s+", _strip_leading_hash(lines[radial_header_idx]))
    radial_cols = _normalize_headers(radial_header_tokens)

    # ---- 4) Parse radial numeric lines until blank or next header block ----
    radial: List[Dict[str, float]] = []
    for ln in lines[radial_header_idx + 1:]:
        s = ln.strip()
        if not s:
            # stop once we've started collecting
            if radial:
                break
            continue
        if s.startswith("#"):
            # stop if a new comment block begins after collecting
            if radial:
                break
            continue

        if not num_re.match(s):
            # stop once numeric block ends (after it started)
            if radial:
                break
            continue

        parts = re.split(r"\s+", s)
        vals = [float(x) for x in parts]
        if len(vals) < len(radial_cols):
            # allow short lines; skip (or pad if you prefer)
            continue
        vals = vals[:len(radial_cols)]
        radial.append({radial_cols[j]: vals[j] for j in range(len(radial_cols))})

    radial = reformat_radial(radial)

    return summary, radial

def _normalize_headers(tokens: List[str]) -> List[str]:
    # e.g. "T(N)" -> "T_N", "Pshaft(W)" -> "Pshaft_W"
    out = []
    for t in tokens:
        t2 = t.replace("(", "_").replace(")", "").replace("/", "_per_")
        t2 = re.sub(r"[^\w]+", "_", t2).strip("_")
        out.append(t2)
    return out

def _strip_leading_hash(line: str) -> str:
    s = line.lstrip()
    if s.startswith("#"):
        return s[1:].lstrip()
    return line.strip()

def reformat_radial(rad_dict):
    if not rad_dict:
        return {}
    
    keys = rad_dict[0].keys()

    out: Dict[str, List[Any]] = {k: [] for k in keys}
    for row in rad_dict:
        for k in keys:
            out[k].append(row[k])

    return out

def _find_summary_row(lines, header_idx, expected_n):
    # Stop scanning once the radial header starts; prevents grabbing radial rows as summary
    for ln in lines[header_idx + 1:]:
        s = ln.strip()
        if not s:
            continue

        # If we reached radial header, stop — summary must appear before this.
        low = s.lower()
        if "radius" in low and "chord" in low and "beta" in low and "cl" in low:
            break

        # In QPROP 1.22 single-point output, the summary numeric line is prefixed by '#'
        if not s.startswith("#"):
            continue

        vals = _extract_floats_with_repair(s)

        # Sometimes you might get extra columns; trim
        if len(vals) >= expected_n:
            return vals[:expected_n]

        # If too short, keep scanning (maybe the line is malformed)
        continue

    raise ValueError("Could not find a valid summary numeric row (before radial table).")

def _extract_floats_with_repair(line: str) -> List[float]:
    """
    Extract floats from a QPROP line, repairing the common Fortran formatting glitch
    where two floats get concatenated like: 0.2328E+051000.023
    """
    s = line.strip()
    if s.startswith("#"):
        s = s[1:].strip()

    tokens = re.split(r"\s+", s)
    out: List[float] = []

    for tok in tokens:
        # normal token parses fine
        if re.fullmatch(_FLOAT, tok):
            out.append(float(tok))
            continue

        # attempt repair for concatenated exponent form:
        # e.g. 0.2328E+051000.023  -> 0.2328E+05  and  1000.023
        m = re.fullmatch(rf"({_FLOAT})(.*)", tok)
        if m:
            first = m.group(1)
            rest = m.group(2)

            # if 'rest' looks like another float (possibly with sign), parse it too
            if rest and re.fullmatch(_FLOAT, rest):
                out.append(float(first))
                out.append(float(rest))
                continue

            # More specific split: split right after exponent part if present
            m2 = re.fullmatch(r"(.+[eE][-+]?\d+)([-+]?\d.*)", tok)
            if m2:
                a, b = m2.group(1), m2.group(2)
                if re.fullmatch(_FLOAT, a) and re.fullmatch(_FLOAT, b):
                    out.append(float(a))
                    out.append(float(b))
                    continue

        # If we get here, token is truly unparsable; ignore it (or raise)
        # I prefer raising so you catch new formats early:
        raise ValueError(f"Could not parse numeric token: {tok}")

    return out




