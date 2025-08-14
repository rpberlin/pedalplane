import os
import re
import time
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Your XFOIL executable
XFOIL_EXE = "/Users/ryanblanchard/myApplications/Xfoil/bin/xfoil"
AIRFOIL_DB = '/Users/ryanblanchard/Documents/pedalplane/airfoils/'
POLARS_DB = '/Users/ryanblanchard/Documents/pedalplane/airfoils/polars/'

def plot_polar_contours(df: pd.DataFrame, nlevels=20, log_x=True, interpolate=False, eps=1e-8):
    """
    Expects df with columns: Re, AoA, CL, CD, CM  (case-sensitive).
    Produces a 2x2 contour plot: CL, CD, CM, and CL/CD vs (Re, AoA).
    """

    # Ensure numeric & sorted unique axes
    Re_vals  = np.array(sorted(df["Re"].astype(float).unique()))
    AoA_vals = np.array(sorted(df["AoA"].astype(float).unique()))

    # Helper: build Z grid by pivoting (AoA rows, Re cols)
    def grid(var):
        pivot = (
            df.pivot(index="AoA", columns="Re", values=var)
              .reindex(index=AoA_vals, columns=Re_vals)
        )
        return pivot.values  # shape (len(AoA_vals), len(Re_vals))

    # Base grids (may contain NaNs if some points failed)
    Z_CL = grid("Cl")
    Z_CD = grid("Cd")
    Z_CM = grid("Cm")

    # Optional: fill gaps by interpolation if requested
    if interpolate and (np.isnan(Z_CL).any() or np.isnan(Z_CD).any() or np.isnan(Z_CM).any()):
        try:
            from scipy.interpolate import griddata
            XY = df[["Re", "AoA"]].astype(float).to_numpy()
            Xg, Yg = np.meshgrid(Re_vals, AoA_vals)
            def fill(varname):
                Z = griddata(XY, df[varname].astype(float).to_numpy(), (Xg, Yg), method="linear")
                # fallback nearest for any remaining NaNs
                if np.isnan(Z).any():
                    Zn = griddata(XY, df[varname].astype(float).to_numpy(), (Xg, Yg), method="nearest")
                    Z = np.where(np.isnan(Z), Zn, Z)
                return Z
            Z_CL, Z_CD, Z_CM = fill("CL"), fill("CD"), fill("CM")
        except ImportError:
            print("scipy not available; skipping interpolation fill.")

    # Ratio (avoid divide-by-zero)
    Z_CLCD = Z_CL / (Z_CD + eps)

    # Mesh for plotting
    X, Y = np.meshgrid(Re_vals, AoA_vals)

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    ax11, ax12, ax21, ax22 = axes.ravel()

    def do_contour(ax, Z, title, xlabel=False, ylabel=False):
        cs = ax.contourf(X, Y, Z, levels=nlevels)
        fig.colorbar(cs, ax=ax)
        ax.set_title(title)
        if log_x:
            ax.set_xscale("log")
        if xlabel:
            ax.set_xlabel("Re")
        if ylabel:
            ax.set_ylabel("AoA [deg]")

    do_contour(ax11, Z_CL,   "CL",    ylabel=True)
    do_contour(ax12, Z_CD,   "CD")
    do_contour(ax21, Z_CM,   "CM",    xlabel=True, ylabel=True)
    do_contour(ax22, Z_CLCD, "CL/CD", xlabel=True)

    plt.show()

def sweep_aoa_re(
    airfoil: str,
    re_values,
    aoa_values,
    ncrit=9,
    mach=0.0,
    max_iter=200,
    retries=(400, 800),   # bump ITER on retries
    debug=False
) -> pd.DataFrame:
    """
    Run a grid sweep over Re × AoA using run_xfoil().
    Returns a DataFrame with columns:
      ['Re','AoA','CL','CD','CM','elapsed_s','status']
    """
    rows = []
    for Re in re_values:
        for AoA in aoa_values:
            result = run_xfoil(airfoil, Re, AoA, Ncrit=ncrit, Mach=mach, max_iter=max_iter, debug=debug)
            status = "ok" if (result.get("Cl") is not None and not np.isnan(result.get("Cl", np.nan))) else "fail"

            # Retry strategy if failed
            if status != "ok":
                for it in retries:
                    retry_res = run_xfoil(airfoil, Re, AoA, Ncrit=ncrit, Mach=mach, max_iter=it, debug=debug)
                    if retry_res.get("Cl") is not None and not np.isnan(retry_res.get("Cl", np.nan)):
                        result, status = retry_res, "ok"
                        break

            # Assemble row with safe defaults
            row = {
                "Re": Re,
                "AoA": AoA,
                "Cl": float(result.get("Cl") or -.01234),
                "Cd": float(result.get('Cd') or 0.0314),
                "Cm": float(result.get('Cm') or 0.0012345),
                "elapsed_s": float(result.get("elapsed_s", np.nan)),
                "status": status,
            }
            rows.append(row)
    return pd.DataFrame(rows)

def save_results(df: pd.DataFrame, airfoil: str,Re_list, AoA_list):
    Re_low_str = '_'+str(Re_list[0]).split('.')[0]
    Re_high_str = '_'+str(Re_list[-1]).split('.')[0]
    AoA_low_str = '_'+str(AoA_list[0]).split('.')[0]
    AoA_high_str = '_'+str(AoA_list[-1]).split('.')[0]
    db_basename = airfoil.split('.')[0] +'_Re'+Re_low_str+Re_high_str+'_AoA'+AoA_low_str+AoA_high_str
    csv_path = POLARS_DB + db_basename+'.csv'
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

def _short_tmp_dir():
    """
    Return a short, safe temp directory path (prefer /tmp on POSIX).
    Falls back to Python's temp dir if needed.
    """
    if os.name == "posix" and os.path.isdir("/tmp"):
        return "/tmp"
    # Fallback (should still be short on mac/linux; on Windows this may be long)
    return os.path.abspath(os.getenv("TMPDIR") or os.getenv("TEMP") or os.getenv("TMP") or "/tmp")

def run_xfoil(
    airfoil,
    Re,
    AoA,
    Ncrit=9,
    Mach=0.0,
    max_iter=200,
    xfoil_path=XFOIL_EXE,
    debug=False
):
    """
    Run XFOIL once at a given Re and AoA, returning Cl, Cd, Cm and elapsed time.
    Writes the polar file to a short, safe path to avoid XFOIL path issues.

    Returns:
        dict: {Cl, Cd, Cm, elapsed_s} on success, or includes {error, reason} on failure.
    """
    airfoil_path = os.path.abspath(AIRFOIL_DB+airfoil)

    # --- Build a short, unique polar path in /tmp ---
    tmpdir = _short_tmp_dir()
    unique = f"{os.getpid()}_{int(time.time()*1000)}"
    polar_file = os.path.join(tmpdir, f"xfoil_{unique}.pol")   # no spaces, very short

    # --- Construct XFOIL command script line-by-line (no stray whitespace) ---
    xfoil_cmds = [
        f"LOAD {airfoil_path}",
        #"PANE",
        "OPER",
        f"VISC {Re:.1f}",
        f"MACH {Mach}",
        f"N {Ncrit}",
        f"ITER {max_iter}",
        "PACC",
        polar_file,   # output filename
        "",           # blank line: start accumulation (no dump file)
        f"ALFA {AoA:.2f}",
        "PACC",       # close accumulation
        "",           # blank line again
        "QUIT"
    ]
    xfoil_input = "\n".join(xfoil_cmds)

    if debug:
        print("=== XFOIL INPUT ===")
        print(xfoil_input)
        print("===================")

    start = time.time()
    try:
        proc = subprocess.run(
            [xfoil_path],
            input=xfoil_input.encode("ascii", errors="ignore"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return {"Cl": None, "Cd": None, "Cm": None, "elapsed_s": time.time()-start,
                "error": "timeout", "reason": "XFOIL did not finish in time."}

    elapsed = time.time() - start
    stdout_text = proc.stdout.decode(errors="ignore")
    stderr_text = proc.stderr.decode(errors="ignore")

    if debug:
        print("=== XFOIL STDOUT (tail) ===")
        print("\n".join(stdout_text.splitlines()[-40:]))
        print("===========================")
        if stderr_text.strip():
            print("=== XFOIL STDERR ===")
            print(stderr_text)
            print("====================")

    # --- Parse polar file, then clean it up ---
    try:
        with open(polar_file, "r") as f:
            lines = f.readlines()
    except Exception:
        # If missing, XFOIL likely rejected the path and defaulted to 'pol'
        # or there was a permissions problem.
        return {"Cl": None, "Cd": None, "Cm": None, "elapsed_s": elapsed,
                "error": "no_polar_file",
                "reason": "Polar file not created. XFOIL may have rejected the path or failed to converge.",
                "stdout_tail": "\n".join(stdout_text.splitlines()[-40:])}

    finally:
        # Best-effort cleanup (safe even if it doesn't exist)
        try:
            if os.path.exists(polar_file):
                os.remove(polar_file)
        except Exception:
            pass

    # Find the last numeric data row
    data_row = None
    for line in reversed(lines):
        if re.match(r"^\s*-?\d", line):
            data_row = line.split()
            break

    if data_row and len(data_row) >= 6:
        try:
            Cl = float(data_row[1])
            Cd = float(data_row[2])
            Cddp = float(data_row[3])
            Cm = float(data_row[4])
            return {"Cl": Cl, "Cd": Cd, "Cm": Cm, "elapsed_s": elapsed}
        except ValueError:
            pass

    # If no numeric row was found, report a helpful error
    return {"Cl": None, "Cd": None, "Cm": None, "elapsed_s": elapsed,
            "error": "no_data",
            "reason": "Polar file had a header but no numeric rows (likely non-convergence or PACC issue).",
            "stdout_tail": "\n".join(stdout_text.splitlines()[-40:])}


if __name__ == "__main__":
    #airfoil = "eppler387.dat"
    airfoil = "dae51.dat"
    
    result = run_xfoil(airfoil, Re=3.4567e5, AoA=5.0, debug=False)
    print(result)

    

    # Choose your grids (tune as needed)
    Re_min = 5000
    Re_max = 1e6
    Re_list  = np.logspace(np.log10(Re_min), np.log10(Re_max), 10)    # ~3.2e4 to 1e6
    AoA_list = np.arange(0, 18, 1)

    df = sweep_aoa_re(airfoil, Re_list, AoA_list, ncrit=9, mach=0.0, max_iter=200, retries=(400, 800), debug=False)
    print(df.head())
    print(df["status"].value_counts())
    plot_polar_contours(df, nlevels=20, log_x=True, interpolate=False, eps=1e-8)
    save_results(df, airfoil, Re_list, AoA_list)