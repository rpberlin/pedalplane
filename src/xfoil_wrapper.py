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
AIRFOIL_DB = '/Users/ryanblanchard/myApplications/Xfoil/airfoils/'
POLARS_DB = '/Users/ryanblanchard/myApplications/Xfoil/airfoils/polars/'

def spaced_indices(M: int, N: int) -> np.ndarray:
    """
    Return N indices for an array of length M, including 0 and M-1,
    spaced as uniformly as possible on the integer grid.

    Requires 1 <= N <= M.
    """
    if N < 1 or N > M:
        raise ValueError("N must satisfy 1 <= N <= M")
    if M == 1 or N == 1:
        return np.array([0], dtype=np.int64)
    k = np.arange(N, dtype=np.int64)
    return (k * (M - 1) // (N - 1)).astype(np.int64)

def plot_polar_contours(df: pd.DataFrame, nlevels=20, log_x=True, interpolate=False, eps=1e-8):
    """
    Expects df with columns: Re, AoA, CL, CD, CM  (case-sensitive).
    Produces a 2x2 contour plot: Cl, Cd, Cm, and CL/CD vs (Re, AoA).
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
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), constrained_layout=True)
    ax11, ax12, ax13, ax21, ax22, ax23 = axes.ravel()

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
    
    Re_list = X[0,:]
    AoA_list = Y[:,0]
    Re_idxs = spaced_indices(len(Re_list),6)
    for Re_idx in Re_idxs:
        Re = Re_list[Re_idx]
        ax13.plot(AoA_list,Z_CL[:,Re_idx], label='Re = '+str(Re))
        ax23.plot(AoA_list,Z_CLCD[:,Re_idx], label='Re = '+str(Re))
    ax13.set_xlabel('Angle of Attack (deg)')
    ax13.set_ylabel('Lift Coefficient (-)')
    ax23.set_xlabel('Angle of Attack (deg)')
    ax23.set_ylabel('Lift/Drag Ratio (-)')
    plt.legend()

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
                "Cl": float(result.get("Cl") or -5),
                "Cd": float(result.get('Cd') or 5),
                "Cm": float(result.get('Cm') or 5),
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

def run_xfoil_naca_with_cpmin(
    naca_code,
    Re,
    AoA,
    Ncrit=9,
    Mach=0.0,
    max_iter=200,
    xfoil_path=XFOIL_EXE,
    debug=False
):
    """
    Run XFOIL once at a given Re and AoA, returning Cl, Cd, Cm, Cp_min, and elapsed time.
    Writes outputs to short, safe paths to avoid XFOIL path issues.

    Returns:
        dict: {Cl, Cd, Cm, Cp_min, Cp_min_x, elapsed_s} on success.
              On failure, fields may be None and include {error, reason}.
    """

    # --- Build short, unique paths in /tmp ---
    tmpdir = _short_tmp_dir()
    unique = f"{os.getpid()}_{int(time.time()*1000)}"
    polar_file = os.path.join(tmpdir, f"xfoil_{unique}.pol")  # very short
    cp_file    = os.path.join(tmpdir, f"xfoil_{unique}.cp")   # very short

    # --- Construct XFOIL command script ---
    xfoil_cmds = [
        f"NACA {naca_code}",
        # "PANE",            # enable if you want repaneling here
        "OPER",
        "PACC",
        polar_file,
        ""
        f"VISC {Re:.1f}",
        f"ITER {max_iter}",
        "CINC",                                       
        f"ALFA {AoA:.2f}", 
                # solve point                       
        f"CPWR {cp_file}",          # write surface Cp distribution for current solution
        "",
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
        return {"Cl": None, "Cd": None, "Cm": None, "Cp_min": None, "Cp_min_x": None,
                "elapsed_s": time.time()-start, "error": "timeout",
                "reason": "XFOIL did not finish in time."}

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

    # --- Parse polar file ---
    try:
        with open(polar_file, "r") as f:
            lines = f.readlines()
    except Exception:
        result = {"Cl": None, "Cd": None, "Cm": None, "Cp_min": None, "Cp_min_x": None,
                  "elapsed_s": elapsed, "error": "no_polar_file",
                  "reason": "Polar file not created. XFOIL may have rejected the path or failed to converge.",
                  "stdout_tail": "\n".join(stdout_text.splitlines()[-40:])}
        # cleanup before return
        try:
            if os.path.exists(polar_file): os.remove(polar_file)
            if os.path.exists(cp_file):    os.remove(cp_file)
        except Exception:
            pass
        return result

    # find last numeric data row
    data_row = None
    for line in reversed(lines):
        if re.match(r"^\s*-?\d", line):
            data_row = line.split()
            break

    Cl = Cd = Cm = None
    if data_row and len(data_row) >= 6:
        try:
            Cl = float(data_row[1])
            Cd = float(data_row[2])
            # Cddp = float(data_row[3])  # available if you need it
            Cm = float(data_row[4])
        except ValueError:
            pass

    # --- Parse Cp file for minimum Cp (most negative) across the surface ---
    Cp_min = None
    Cp_min_x = None
    try:
        with open(cp_file, "r") as f:
            for line in f:
                if not re.match(r"^\s*-?\d", line):  # skip headers/blank lines
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                # XFOIL CPWR formats vary slightly (x Cp) or (x y Cp). Take last token as Cp and first as x.
                try:
                    x_val = float(parts[0])
                    cp_val = float(parts[-1])
                except ValueError:
                    continue
                if Cp_min is None or cp_val < Cp_min:
                    Cp_min = cp_val
                    Cp_min_x = x_val
    except Exception:
        # leave Cp_min as None if file missing or unreadable
        pass
    finally:
        # cleanup temp files
        try:
            if os.path.exists(polar_file): os.remove(polar_file)
            if os.path.exists(cp_file):    os.remove(cp_file)
        except Exception:
            pass

    if Cl is not None and Cd is not None and Cm is not None:
        return {"Cl": Cl, "Cd": Cd, "Cm": Cm, "Cp_min": Cp_min, "Cp_min_x": Cp_min_x, "elapsed_s": elapsed}

    # No numeric row found
    return {"Cl": None, "Cd": None, "Cm": None, "Cp_min": Cp_min, "Cp_min_x": Cp_min_x,
            "elapsed_s": elapsed, "error": "no_data",
            "reason": "Polar file had a header but no numeric rows (likely non-convergence or PACC issue).",
            "stdout_tail": "\n".join(stdout_text.splitlines()[-40:])}

def run_xfoil_with_cpmin(
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
    Run XFOIL once at a given Re and AoA, returning Cl, Cd, Cm, Cp_min, and elapsed time.
    Writes outputs to short, safe paths to avoid XFOIL path issues.

    Returns:
        dict: {Cl, Cd, Cm, Cp_min, Cp_min_x, elapsed_s} on success.
              On failure, fields may be None and include {error, reason}.
    """
    airfoil_path = os.path.abspath(AIRFOIL_DB + airfoil)

    # --- Build short, unique paths in /tmp ---
    tmpdir = _short_tmp_dir()
    unique = f"{os.getpid()}_{int(time.time()*1000)}"
    polar_file = os.path.join(tmpdir, f"xfoil_{unique}.pol")  # very short
    cp_file    = os.path.join(tmpdir, f"xfoil_{unique}.cp")   # very short

    # --- Construct XFOIL command script ---
    xfoil_cmds = [
        f"LOAD {airfoil_path}",
        # "PANE",            # enable if you want repaneling here
        "OPER",
        "PACC",
        polar_file,
        ""
        f"VISC {Re:.1f}",
        f"ITER {max_iter}",
        "CINC",                                       
        f"ALFA {AoA:.2f}", 
                # solve point                       
        f"CPWR {cp_file}",          # write surface Cp distribution for current solution
        "",
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
        return {"Cl": None, "Cd": None, "Cm": None, "Cp_min": None, "Cp_min_x": None,
                "elapsed_s": time.time()-start, "error": "timeout",
                "reason": "XFOIL did not finish in time."}

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

    # --- Parse polar file ---
    try:
        with open(polar_file, "r") as f:
            lines = f.readlines()
    except Exception:
        result = {"Cl": None, "Cd": None, "Cm": None, "Cp_min": None, "Cp_min_x": None,
                  "elapsed_s": elapsed, "error": "no_polar_file",
                  "reason": "Polar file not created. XFOIL may have rejected the path or failed to converge.",
                  "stdout_tail": "\n".join(stdout_text.splitlines()[-40:])}
        # cleanup before return
        try:
            if os.path.exists(polar_file): os.remove(polar_file)
            if os.path.exists(cp_file):    os.remove(cp_file)
        except Exception:
            pass
        return result

    # find last numeric data row
    data_row = None
    for line in reversed(lines):
        if re.match(r"^\s*-?\d", line):
            data_row = line.split()
            break

    Cl = Cd = Cm = None
    if data_row and len(data_row) >= 6:
        try:
            Cl = float(data_row[1])
            Cd = float(data_row[2])
            # Cddp = float(data_row[3])  # available if you need it
            Cm = float(data_row[4])
        except ValueError:
            pass

    # --- Parse Cp file for minimum Cp (most negative) across the surface ---
    Cp_min = None
    Cp_min_x = None
    try:
        with open(cp_file, "r") as f:
            for line in f:
                if not re.match(r"^\s*-?\d", line):  # skip headers/blank lines
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                # XFOIL CPWR formats vary slightly (x Cp) or (x y Cp). Take last token as Cp and first as x.
                try:
                    x_val = float(parts[0])
                    cp_val = float(parts[-1])
                except ValueError:
                    continue
                if Cp_min is None or cp_val < Cp_min:
                    Cp_min = cp_val
                    Cp_min_x = x_val
    except Exception:
        # leave Cp_min as None if file missing or unreadable
        pass
    finally:
        # cleanup temp files
        try:
            if os.path.exists(polar_file): os.remove(polar_file)
            if os.path.exists(cp_file):    os.remove(cp_file)
        except Exception:
            pass

    if Cl is not None and Cd is not None and Cm is not None:
        return {"Cl": Cl, "Cd": Cd, "Cm": Cm, "Cp_min": Cp_min, "Cp_min_x": Cp_min_x, "elapsed_s": elapsed}

    # No numeric row found
    return {"Cl": None, "Cd": None, "Cm": None, "Cp_min": Cp_min, "Cp_min_x": Cp_min_x,
            "elapsed_s": elapsed, "error": "no_data",
            "reason": "Polar file had a header but no numeric rows (likely non-convergence or PACC issue).",
            "stdout_tail": "\n".join(stdout_text.splitlines()[-40:])}

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

def df_cleanup(df: pd.DataFrame, value_cols=("Cl", "Cd", "Cm")) -> pd.DataFrame:
    """
    Clean up spurious interior outliers in a Re×AoA grid for the given value columns.
    A point is replaced if it's NOT on the edge and its value lies more than
    2× outside the [min, max] range of its four immediate neighbors (up/down/left/right).
    Replacement value is the median of those neighbors (requires ≥3 valid neighbors).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'Re' and 'AoA' and at least the columns in value_cols.
    value_cols : tuple[str]
        Which columns to clean (default: ('CL','CD','CM')).

    Returns
    -------
    df_clean : pd.DataFrame
        Copy of df with cleaned values in the specified columns.
    """
    if not {"Re", "AoA"}.issubset(df.columns):
        raise ValueError("df must contain 'Re' and 'AoA' columns")

    df_clean = df.copy()

    # Sorted unique axes (ensure numeric)
    Re_vals  = np.array(sorted(df_clean["Re"].astype(float).unique()))
    AoA_vals = np.array(sorted(df_clean["AoA"].astype(float).unique()))
    nA, nR = len(AoA_vals), len(Re_vals)

    # Helper to push a cleaned grid back into the long dataframe for one column
    def _write_back(col, Z_new):
        upd = (
            pd.DataFrame(Z_new, index=AoA_vals, columns=Re_vals)
              .stack()
              .rename(col)
              .reset_index()
              .rename(columns={"level_0": "AoA", "level_1": "Re"})
        )
        # Merge back just this column (preserve all other columns as-is)
        df_clean[col] = pd.merge(
            df_clean.drop(columns=[col]),
            upd,
            on=["AoA", "Re"],
            how="left"
        )[col]

    for col in value_cols:
        if col not in df_clean.columns:
            continue  # silently skip missing columns

        # Build grid Z (AoA rows × Re cols)
        pivot = (
            df_clean.pivot(index="AoA", columns="Re", values=col)
                    .reindex(index=AoA_vals, columns=Re_vals)
        )
        Z = pivot.values.astype(float, copy=True)

        # Iterate interior points only
        for i in range(1, nA - 1):
            for j in range(1, nR - 1):
                v = Z[i, j]
                if not np.isfinite(v):
                    continue  # nothing to do if center is NaN/inf

                neighbors = [
                    Z[i-1, j],  # up
                    Z[i+1, j],  # down
                    Z[i, j-1],  # left
                    Z[i, j+1],  # right
                ]
                nb = np.array([x for x in neighbors if np.isfinite(x)], dtype=float)

                if nb.size < 3:
                    # need at least 3 neighbors to make a stable decision
                    continue

                nb_min, nb_max = float(np.min(nb)), float(np.max(nb))
                span = nb_max - nb_min

                # Outlier test: more than 2× outside neighbor range
                lower = nb_min - 0.5 * span
                upper = nb_max + 0.5 * span

                if (v < lower) or (v > upper):
                    # Replace with neighbor median
                    Z[i, j] = float(np.mean(nb))

        # Write cleaned grid back for this column
        _write_back(col, Z)

    return df_clean

def get_qprop_fit_params(airfoil, Re, makeplot = False):
    Re_fit_multiple = 10
    alpha0 = 0 
    alpha1 = 12
    alpha2 = -5
    alphalow = -10
    alphahigh = 20
    alphamid = 0.5*(alpha0+alpha1)
    alphamid2 = 0.5*(alpha0+alpha2)
    result0 = run_xfoil(airfoil,Re,alpha0)
    result1 = run_xfoil(airfoil,Re,alpha1)
    result2 = run_xfoil(airfoil,Re,alpha2)
    resultlow = run_xfoil(airfoil,Re,alphalow)
    resulthigh = run_xfoil(airfoil,Re,alphahigh)
    resultmid = run_xfoil(airfoil,Re,alphamid)
    resultmid2 = run_xfoil(airfoil,Re, alphamid2)

    alpha_list1 = []
    alpha_list2 = []
    Cl_list1 = []
    Cl_list2 = []
    Cd_list1 = []
    Cd_list2 = []

    for tmp_alpha in np.linspace(alpha0,alphahigh,20):
        tmp_res = run_xfoil(airfoil,Re, tmp_alpha)
        Cl_tmp, Cd_tmp = tmp_res['Cl'], tmp_res['Cd']
        if Cl_tmp is not None and Cd_tmp is not None:
            alpha_list1.append(tmp_alpha)
            Cl_list1.append(Cl_tmp)
            Cd_list1.append(Cd_tmp)

    for tmp_alpha in np.linspace(alphalow,alpha0,10):
        tmp_res = run_xfoil(airfoil,Re, tmp_alpha)
        Cl_tmp, Cd_tmp = tmp_res['Cl'], tmp_res['Cd']
        if Cl_tmp is not None and Cd_tmp is not None:
            alpha_list2.append(tmp_alpha)
            Cl_list2.append(Cl_tmp)
            Cd_list2.append(Cd_tmp)

    if len(alpha_list1) < 5 or len(alpha_list2) < 3:
        return None
    

    alpha_list1 = np.asarray(alpha_list1)
    i0 = np.argmin(np.abs(alpha_list1 - alpha0))
    i1 = np.argmin(np.abs(alpha_list1 - alpha1))
    ihigh = np.argmin(np.abs(alpha_list1 - alphahigh))
    imid = np.argmin(np.abs(alpha_list1 - alphamid))
    alpha0 ,Cl0, Cd0 = alpha_list1[i0], Cl_list1[i0], Cd_list1[i0]
    alpha1 ,Cl1, Cd1 = alpha_list1[i1], Cl_list1[i1], Cd_list1[i1]
    alphahigh ,Clhigh, Cdhigh = alpha_list1[ihigh], Cl_list1[ihigh], Cd_list1[ihigh]
    alphamid ,Clmid, Cdmid = alpha_list1[imid], Cl_list1[imid], Cd_list1[imid]

    resultRefit = run_xfoil(airfoil,Re_fit_multiple*Re,alpha0)
    ClRefit, CdRefit = resultRefit['Cl'], resultRefit['Cd']

    alpha_list2 = np.asarray(alpha_list2)
    i2 = np.argmin(np.abs(alpha_list2 - alpha2))
    ilow = np.argmin(np.abs(alpha_list2 - alphalow))
    imid2 = np.argmin(np.abs(alpha_list2 - alphamid2))
    alpha2 ,Cl2, Cd2 = alpha_list2[i2], Cl_list2[i2], Cd_list2[i2]
    alphalow ,Cllow, Cdlow = alpha_list2[ilow], Cl_list2[ilow], Cd_list2[ilow]
    alphamid2 ,Clmid2, Cdmid2 = alpha_list2[imid2], Cl_list2[imid2], Cd_list2[imid2]

    
    #Cl0, Cd0 = result0['Cl'], result0['Cd']
    #Cl1, Cd1 = result1['Cl'], result1['Cd']
    #Cl2, Cd2 = result2['Cl'], result2['Cd']
    #Cllow, Cdlow = resultlow['Cl'], resultlow['Cd']
    #Clhigh, Cdhigh = resulthigh['Cl'], resulthigh['Cd']
    #Clmid, Cdmid = resultmid['Cl'], resultmid['Cd']
    #Clmid2, Cdmid2 = resultmid2['Cl'], resultmid2['Cd']
    
    Cl_a_deg = (Cl1-Cl0)/(alpha1-alpha0)
    Cl_a_rad = (180/3.14159)*Cl_a_deg
    #Clx = [Cl0,Clmid,Cl1]
    #Cdy = [Cd0,Cdmid,Cd1]
    #Clx2 = [Cl2,Clmid2,Cl0]
    #Cdy2 = [Cd2,Cdmid2,Cd0]
    Clx = Cl_list1[i0:i1+1]
    Cdy = Cd_list1[i0:i1+1]
    Clx2 = Cl_list2[i2:-1]
    Cdy2 = Cd_list2[i2:-1]

    parab_terms = np.polyfit(Clx, Cdy, 2)
    parab_terms2 = np.polyfit(Clx2, Cdy2, 2)
    [Cd2u, b, c] = parab_terms
    [Cd2l, b2,c2] = parab_terms2
    fitted_parab = np.poly1d(parab_terms)
    fitted_parab2 = np.poly1d(parab_terms2)

    Re_exp = np.log(CdRefit/Cd0)/np.log(Re_fit_multiple)

    prop_params = {}
    prop_params['CL0'] = Cl0
    prop_params['CL_a'] = Cl_a_rad
    prop_params['CD0'] = Cd0
    prop_params['Cd2u'] = max(Cd2u,0.001)
    prop_params['Cd2l'] = max(Cd2l,0.001)
    prop_params['CLCD0'] = Cl0
    prop_params['Re_ref'] = Re
    prop_params['Re_exp'] = min(Re_exp,0)
    prop_params['Cl_min'] = min(Cl_list2)
    prop_params['Cl_max'] = max(Cl_list1)

    print('\nParameters for Qprop: ', airfoil)
    for key in prop_params:
        print(key, prop_params[key])


    print(Cl_a_rad)
    if makeplot == True:
        alpha_pts = np.linspace(alphalow,alphahigh,20)
        Cl_theory = 6.28*(3.14/180)*(alpha_pts-alpha0)+Cl0
        Cl_fit = Cl_a_deg*(alpha_pts-alpha0)+Cl0
        Cl_pts = [ Cl0+Cl_a_deg*(alpha_tmp-alpha0) for alpha_tmp in alpha_pts]
        
        Cd_pts = fitted_parab(Cl_pts)
        Cd_pts2 = fitted_parab2(Cl_pts)

        plt.plot([alphalow,alpha2,alphamid2,alpha0,alphamid,alpha1,alphahigh],[Cllow,Cl2, Clmid2, Cl0,Clmid,Cl1,Clhigh],'o')
        plt.plot(alpha_pts,Cl_theory,label='2pi-Theory')
        plt.plot(alpha_pts,Cl_fit,label='Current Fit')
        plt.title(airfoil)
        plt.xlabel('Angle of Attack (deg)')
        plt.ylabel('Lift Coefficient (-)')
        plt.legend()
        plt.show()

        plt.plot([Cdlow,Cd2, Cdmid2, Cd0,Cdmid,Cd1,Cdhigh],[Cllow,Cl2, Clmid2, Cl0,Clmid,Cl1,Clhigh],'o')
        plt.title(airfoil)
        plt.plot(Cdy,Clx,'+',label='to fit1')
        plt.plot(Cdy2,Clx2,'x',label='to fit2')
        plt.plot(Cd_pts,Cl_pts,label='Cd2u')
        plt.plot(Cd_pts2,Cl_pts,label="Cd2l")
        plt.xlabel('Drag Coefficient(-)')
        plt.ylabel('Lift Coefficient (-)')
        plt.legend()
        plt.show()

    return prop_params  


if __name__ == "__main__":
    #airfoil = "eppler387.dat"
    airfoil = "dae11.dat"
    naca_code = 2315
    
    #result = run_xfoil(airfoil, Re=3.4567e5, AoA=5.0, debug=True)
    #resultw = run_xfoil_with_cpmin(airfoil, Re=1e6, AoA=5.0, debug=True)
    Cpmins = []
    Clifts = []
    #naca_codes = range(2313,2318)
    #for naca_code in naca_codes:
    #    resultn = run_xfoil_naca_with_cpmin(naca_code, Re=3.4567e5, AoA=5.0, debug=True)
    #    Cpmins.append(resultn['Cp_min'])
    #    Clifts.append(resultn['Cl'])
    #plt.plot(naca_codes,Cpmins)
    #plt.show()
    #plt.plot(naca_codes,Clifts)
        #print(resultn)
    #result = run_xfoil(airfoil, Re=3.4567e5, AoA=5.0, debug=True)

    # Choose your grids (tune as needed)

    #result = run_xfoil('dae11.dat', 1e5, 1)
    
    
    Re_min = 3e5
    Re_max = 2e6
    Re_list  = np.logspace(np.log10(Re_min), np.log10(Re_max), 6)    # ~3.2e4 to 1e6
    AoA_list = np.arange(0, 16, 1.0)



    df = sweep_aoa_re(airfoil, Re_list, AoA_list, ncrit=9, mach=0.0, max_iter=200, retries=(400, 800), debug=False)
    print(df.head())
    print(df["status"].value_counts())
    df_clean = df_cleanup(df)
    plot_polar_contours(df_clean, nlevels=20, log_x=True, interpolate=False, eps=.05)
    save_results(df_clean, airfoil, Re_list, AoA_list)