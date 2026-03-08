"""
sensitivity_curves.py
=====================
Generates Figure 3 for the quantum decoherence DM detector paper:
  - Sensitivity reach in the (m_DM, d_e) plane from the Cramér-Rao bound
  - Theoretical exclusion overlays from AxionLimits
  - All data is computed analytically from the equations in the paper

Equations used (all from main.tex):
  - d_e_min : direct linear coupling (Eq. 5)
               delta_omega/omega = kappa_e * d_e * (phi0/M_Pl) * cos(omega_DM*t)
               Accumulated Ramsey phase: Delta_phi = kappa_e * d_e * (phi0/M_Pl) * omega_Q * T
               Threshold: sigma_C / sqrt(N_eff)
               Matches AxionLimits d_e normalization (Derevianko 2018, PRD 97, 042506)
  - phi_0   : Eq. (2.2)  phi_0 = sqrt(2*rho_DM) / m_DM  [natural units, eV]
  - tau_c   : Appendix D  tau_c ~ hbar / (m_DM * v0^2)
  - Fisher  : Eq. (7.1)  I(g) = sum_i (1/sigma_i^2)(d C/d g)^2
  - C-R bound: Eq. (7.2) Delta_g >= 1/sqrt(I(g))

Physical constants (natural units where hbar = c = 1):
  rho_DM = 0.4 GeV/cm^3  (local DM density)
  v0     = 220 km/s       (virial velocity)
  hbar   = 6.582e-16 eV·s
  M_Pl   = 2.435e27 eV    (reduced Planck mass)

CHANGES FROM PREVIOUS VERSION:
  - Replaced g_min_CR (bath-mediated g^4 scaling, units eV^-1) with de_min
    (direct linear d_e coupling, dimensionless), matching AxionLimits convention.
  - rho_DM now computed in natural units (eV^4) via hbar*c conversion.
  - Incoherent regime correction (T > tau_c) is now handled inside de_min.
  - y-axis label corrected to d_e [dimensionless].
  - Updated abstract numbers: d_e < 5e-15 at 1e-22 eV, d_e < 5e-11 at 1e-18 eV
    (T=1yr, N=1e6).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from scipy.special import erf

# ── matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex":         False,         # set True if LaTeX is installed
    "font.family":         "serif",
    "font.size":           11,
    "axes.labelsize":      13,
    "axes.titlesize":      13,
    "legend.fontsize":     9.5,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.top":           True,
    "ytick.right":         True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "figure.dpi":          150,
})

# ── Physical constants ─────────────────────────────────────────────────────────
hbar_eVs   = 6.582119569e-16    # hbar in eV·s
c_ms       = 2.99792458e8       # speed of light m/s
c_kms      = c_ms / 1e3         # speed of light km/s
v0_kms     = 220.0              # DM virial velocity km/s
v0         = v0_kms / c_kms     # dimensionless (v/c)

# rho_DM in natural units (eV^4):
# 0.4 GeV/cm^3 converted via hbar*c = 197.3 MeV*fm = 197.3e-13 MeV*cm
# rho [eV^4] = rho [MeV/cm^3] * (hbar*c [MeV*cm])^{-3}  ... in natural units
_rho_MeV4  = 0.4e3 * (197.3e-13)**3   # MeV^4
rho_DM_eV4 = _rho_MeV4 * 1e24         # eV^4  (1 MeV = 1e6 eV, so MeV^4 -> eV^4: *1e24)

# Qubit and coupling parameters
M_Pl_eV    = 2.435e27                  # reduced Planck mass [eV]
omega_Q_eV = 2 * np.pi * 1e9 * hbar_eVs  # qubit frequency 1 GHz in [eV]
kappa_e    = 1.0                        # dimensionless sensitivity coefficient (Eq. 5)

# ── DM field amplitude (Eq. 2.2, natural units) ───────────────────────────────
def phi0_nat(m_eV):
    """
    phi_0 = sqrt(2 * rho_DM) / m_DM   in natural units [eV].
    Dimensionless amplitude is phi_0 / M_Pl.
    """
    return np.sqrt(2.0 * rho_DM_eV4) / m_eV

# ── DM coherence time (Appendix D) ────────────────────────────────────────────
def tau_c(m_eV):
    """
    tau_c ~ hbar / (m_DM * v0^2)   [seconds]
    Sets the maximum useful coherent integration time for a given DM mass.
    """
    omega_DM = m_eV / hbar_eVs     # rad/s
    return 1.0 / (omega_DM * v0**2)

# ── Cramér-Rao d_e sensitivity (replaces g_min_CR) ───────────────────────────
def de_min(m_eV, T_s, N_runs, sigma_C=0.01, gamma_bath=1e3):
    """
    Minimum detectable scalar-EM coupling d_e from the Cramér-Rao bound.

    Physical model (Eq. 5 of main text):
      delta_omega / omega_Q = kappa_e * d_e * (phi0 / M_Pl) * cos(omega_DM * t)

    Accumulated Ramsey phase at CPMG resonance over effective time T_eff:
      Delta_phi = kappa_e * d_e * (phi0 / M_Pl) * omega_Q [eV] * T_eff [eV^{-1}]

    Detection threshold (shot-noise limited):
      Delta_phi > sigma_C / sqrt(N_eff)

    Coherence regime:
      - T_s <= tau_c(m):  coherent regime, T_eff = T_s, N_eff = N_runs
      - T_s >  tau_c(m):  incoherent regime, average over N_coh = T_s/tau_c windows,
                          T_eff = tau_c, N_eff = N_runs * (T_s / tau_c)

    Matches the AxionLimits d_e normalization (Derevianko PRD 2018, Eq. 4).

    Parameters
    ----------
    m_eV      : DM mass [eV]
    T_s       : total integration time per run [s]
    N_runs    : number of independent measurement runs
    sigma_C   : coherence measurement uncertainty (default 0.01)
    gamma_bath: bath cutoff frequency [rad/s] — unused in this formula but
                retained for API compatibility with the old g_min_CR signature.

    Returns
    -------
    d_e_min   : minimum detectable coupling [dimensionless]
    """
    p0       = phi0_nat(m_eV) / M_Pl_eV    # dimensionless DM field amplitude
    tau      = tau_c(m_eV)                  # DM coherence time [s]

    # Effective coherent integration time and effective shot count
    if T_s > tau:
        T_eff = tau
        N_eff = N_runs * (T_s / tau)        # stack incoherent windows
    else:
        T_eff = T_s
        N_eff = N_runs

    T_eff_eV = T_eff / hbar_eVs             # convert seconds -> eV^{-1}

    # d_e_min from Delta_phi threshold
    return sigma_C / (kappa_e * p0 * omega_Q_eV * T_eff_eV * np.sqrt(N_eff))

# ── Real exclusion data loader ────────────────────────────────────────────────
# Data files from AxionLimits (github.com/cajohare/AxionLimits)
# Download the files listed below into a subfolder: limit_data/ScalarPhoton/
# The coupling in these files is d_e (scalar-EM, dimensionless).
# All files are two columns: mass [eV], d_e [dimensionless]

LIMIT_DATA = r"C:\Users\mitch\OneDrive\Documentos\dm_paper_figures\limit_data\ScalarPhoton"

def load_limit(filename):
    """Load a two-column (mass [eV], coupling) limit file. Returns (m, g) arrays."""
    try:
        dat = np.loadtxt(filename)
        return dat[:, 0], dat[:, 1]
    except FileNotFoundError:
        print(f"  WARNING: {filename} not found — skipping this limit.")
        return None, None

# ── Main plotting routine ─────────────────────────────────────────────────────
def make_figure():

    fig, ax = plt.subplots(figsize=(8, 6))

    # Mass axis: 10^-22 to 10^-6 eV
    m_arr = np.logspace(-22, -6, 600)

    # ── Our sensitivity reach curves ──────────────────────────────────────────
    integration_params = [
        (1.0,        1e3,   "#2166ac", "-",  r"$T=1\,$s,  $N=10^3$"),
        (3600.0,     1e4,   "#4dac26", "-",  r"$T=1\,$hr, $N=10^4$"),
        (86400.0,    1e5,   "#d6604d", "-",  r"$T=1\,$day,$N=10^5$"),
        (3.156e7,    1e6,   "#762a83", "-",  r"$T=1\,$yr, $N=10^6$"),
    ]

    for T_s, N, color, ls, label in integration_params:
        # de_min handles coherent/incoherent transition internally
        d_reach = np.array([de_min(m, T_s, N) for m in m_arr])
        ax.loglog(m_arr, d_reach, color=color, lw=2.0, ls=ls, label=label, zorder=4)

    # ── Existing exclusions and projections (AxionLimits data) ───────────────
    limits = [
        # ── Current exclusions (solid fill) ──
        (f"{LIMIT_DATA}/EotWashEP.txt",
         "Eöt-Wash EP",         "#8c510a", "--",  True),
        (f"{LIMIT_DATA}/MICROSCOPE.txt",
         "MICROSCOPE",          "#bf812d", "--",  True),
        (f"{LIMIT_DATA}/FifthForce.txt",
         "Fifth force",         "#a6611a", "-.",  True),
        (f"{LIMIT_DATA}/GlobularClusters.txt",
         "Glob. clusters",      "#525252", "-.",  True),
        (f"{LIMIT_DATA}/GEO600.txt",
         "GEO600",              "#4d9221", "--",  True),
        (f"{LIMIT_DATA}/LIGO.txt",
         "LIGO",                "#276419", "--",  True),
        (f"{LIMIT_DATA}/Holometer.txt",
         "Holometer",           "#74c476", "--",  True),
        (f"{LIMIT_DATA}/DynamicDecoupling.txt",
         "Dyn. decoupling",     "#fd8d3c", "--",  True),
        (f"{LIMIT_DATA}/AURIGA.txt",
         "AURIGA",              "#d94801", "--",  True),
        (f"{LIMIT_DATA}/DyDy.txt",
         "Dy/Dy clock",         "#6baed6", "--",  True),
        # ── Projections (unfilled, dashed) ──
        (f"{LIMIT_DATA}/Projections/AION-100.txt",
         "AION (proj.)",        "#542788", ":",   False),
        (f"{LIMIT_DATA}/Projections/AEDGE.txt",
         "AEDGE (proj.)",       "#2166ac", ":",   False),
        (f"{LIMIT_DATA}/Projections/MAGIS-100.txt",
         "MAGIS-100 (proj.)",   "#e31a1c", ":",   False),
    ]

    for fpath, label, color, ls, filled in limits:
        m_dat, g_dat = load_limit(fpath)
        if m_dat is None:
            continue
        mask = (m_dat >= 1e-22) & (m_dat <= 1e-6) & (g_dat > 0)
        m_pl, g_pl = m_dat[mask], g_dat[mask]
        if len(m_pl) < 2:
            continue
        if filled:
            ax.fill_between(m_pl, g_pl, 1e0,
                            alpha=0.15, color=color, zorder=2)
        ax.loglog(m_pl, g_pl, color=color, lw=1.5, ls=ls,
                  zorder=3, label=label)

    # ── Axes, labels, limits ───────────────────────────────────────────────────
    ax.set_xlim(1e-22, 1e-6)
    ax.set_ylim(1e-20, 1e-2)
    ax.set_xlabel(r"Dark matter mass $m_\chi$ [eV]")
    ax.set_ylabel(r"Coupling $d_e$ [dimensionless]")
    ax.set_title("Sensitivity reach: fluxonium decoherence spectrometer\n"
                 r"(scalar DM, $d_e$ coupling, $^{13}$C spin bath, CPMG protocol)",
                 pad=10)

    # ── Mass labels on top axis ───────────────────────────────────────────────
    ax2 = ax.twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(ax.get_xlim())
    def eV_to_Hz(m):
        return m / (2 * np.pi * hbar_eVs)
    tick_eV  = np.array([1e-22, 1e-20, 1e-18, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6])
    tick_Hz  = eV_to_Hz(tick_eV)
    ax2.set_xticks(tick_eV)
    ax2.set_xticklabels([f"$10^{{{int(np.log10(f)):.0f}}}$" for f in tick_Hz],
                        fontsize=8)
    ax2.set_xlabel(r"DM oscillation frequency [Hz]", labelpad=6, fontsize=10)

    # ── Annotate mass windows ─────────────────────────────────────────────────
    ax.axvspan(1e-22, 1e-18, alpha=0.04, color="blue")
    ax.text(3e-21, 5e-5, "FDM\nwindow", fontsize=7, color="royalblue",
            ha="center", va="top")
    ax.axvspan(1e-12, 1e-9, alpha=0.04, color="green")
    ax.text(3e-11, 5e-5, "QCD\naxion\nwindow", fontsize=7, color="darkgreen",
            ha="center", va="top")

    # ── Legend ────────────────────────────────────────────────────────────────
    handles, labels = ax.get_legend_handles_labels()
    our_h   = handles[:4];  our_l   = labels[:4]
    exist_h = handles[4:];  exist_l = labels[4:]

    leg1 = ax.legend(our_h, our_l, loc="lower left",
                     title="This work (fluxonium array)",
                     title_fontsize=9, framealpha=0.9, fontsize=9)
    ax.add_artist(leg1)
    ax.legend(exist_h, exist_l, loc="lower right",
              title="Existing / projected",
              title_fontsize=9, framealpha=0.9, fontsize=9,
              ncol=2)

    # ── Footnote ──────────────────────────────────────────────────────────────
    fig.text(0.5, 0.01,
             r"$\rho_\mathrm{DM}=0.4\,\mathrm{GeV/cm}^3$, "
             r"$v_0=220\,\mathrm{km/s}$, "
             r"$\gamma_\mathrm{bath}=10^3\,\mathrm{rad/s}$, "
             r"$\sigma_C=0.01$  —  Exclusions from AxionLimits (cajohare/AxionLimits)",
             ha="center", fontsize=7.5, color="gray")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    return fig

# ── Memory kernel figure (Figure 2) ──────────────────────────────────────────
def make_memory_kernel_figure():
    """
    Plot K(Delta_t) for Markovian (300 K) vs non-Markovian (10 mK).
    Uses the Drude-Lorentz bath (Eq. B.3) and the analytical kernel (Eq. 4.2).
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    t_arr  = np.linspace(0, 5e-3, 2000)   # 0 to 5 ms

    # Drude-Lorentz parameters
    lam    = 1e4    # coupling strength [rad/s]
    gamma  = 1e3    # bath cutoff [rad/s]
    kB     = 8.617e-5  # eV/K

    def kernel_real(t, T_K):
        """
        Real part of K(t): Re[K(t)] = lambda*gamma * coth(gamma/(2 k_B T)) * exp(-gamma*t)
        (leading Drude-Lorentz term, k=0 only)
        """
        if T_K > 0:
            x   = (hbar_eVs * gamma) / (2 * kB * T_K)
            cth = 1.0 / np.tanh(x) if x < 500 else 1.0
        else:
            cth = 1.0
        return lam * gamma * cth * np.exp(-gamma * t)

    def kernel_imag(t):
        """ Imaginary part (temperature-independent for Drude): -lambda*gamma*exp(-gamma*t) """
        return -lam * gamma * np.exp(-gamma * t)

    # Panel 1: Real part at two temperatures
    ax = axes[0]
    colors = {"300 K (Markovian)": ("#d73027", 300.0),
              "10 mK (non-Markovian)": ("#4575b4", 0.010)}

    for label, (color, T_K) in colors.items():
        K_re = np.array([kernel_real(t, T_K) for t in t_arr])
        ax.plot(t_arr * 1e3, K_re / K_re[0], color=color, lw=2.0, label=label)

    # Mark tau_c for 10 mK case
    tau_c_mK = 1.0 / gamma
    ax.axvline(tau_c_mK * 1e3, color="#4575b4", lw=1.0, ls="--", alpha=0.6)
    ax.text(tau_c_mK * 1e3 + 0.05, 0.5, r"$\tau_c = \gamma^{-1}$",
            color="#4575b4", fontsize=9)

    ax.set_xlabel(r"Time delay $\Delta t$ [ms]")
    ax.set_ylabel(r"$\mathrm{Re}[\mathcal{K}(\Delta t)] / \mathcal{K}(0)$")
    ax.set_title("Memory kernel: real part")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color="black", lw=0.5, ls=":")

    # Panel 2: spectral density J(omega) with and without DM perturbation
    ax2  = axes[1]
    omega_arr = np.logspace(0, 6, 1000)

    def J0(omega):
        return 2 * lam * gamma * omega / (omega**2 + gamma**2)

    # DM perturbation at omega_DM = pi/tau_opt ~ 3.14e3 rad/s (example)
    omega_DM = np.pi * gamma
    g_chi_ex = 1e-14   # example coupling
    phi_ex   = phi0_nat(omega_DM * hbar_eVs)
    eps_ex   = g_chi_ex * phi_ex * gamma / (2 * omega_DM)

    def J_DM(omega):
        j0 = J0(omega)
        dj = eps_ex * (J0(omega - omega_DM) - J0(omega + omega_DM))
        return j0, j0 + dj

    J_bare   = J0(omega_arr)
    J_b, J_p = zip(*[J_DM(w) for w in omega_arr])
    J_bare_arr = np.array(J_b)
    J_pert_arr = np.array(J_p)

    ax2.semilogx(omega_arr, J_bare_arr / np.max(J_bare_arr),
                 color="black", lw=1.8, label=r"$J_0(\omega)$ (no DM)")
    ax2.semilogx(omega_arr, J_pert_arr / np.max(J_bare_arr),
                 color="#d73027", lw=1.8, ls="--",
                 label=r"$J_0+\delta J_\mathrm{DM}$ (with DM)")

    ax2.axvline(gamma, color="gray", lw=1.0, ls=":", alpha=0.7)
    ax2.text(gamma * 1.15, 0.55, r"$\gamma$", color="gray", fontsize=9)
    ax2.axvline(omega_DM, color="#d73027", lw=1.0, ls=":", alpha=0.5)
    ax2.text(omega_DM * 1.1, 0.35, r"$\omega_\mathrm{DM}$",
             color="#d73027", fontsize=9)

    ax2.set_xlabel(r"Frequency $\omega$ [rad/s]")
    ax2.set_ylabel(r"$J(\omega)$ / max $J_0$")
    ax2.set_title("Bath spectral density\n" + r"(Drude--Lorentz + DM sideband)")
    ax2.legend(fontsize=9)
    ax2.set_xlim(1, 1e6)
    ax2.set_ylim(-0.05, 1.1)

    plt.tight_layout()
    return fig

# ── ROC curve placeholder (Figure 4) ─────────────────────────────────────────
def make_roc_placeholder():
    """
    Placeholder ROC curve using a simple analytical approximation.
    Real ROC must come from the trained GNN.
    Uses: TPR ~ erf(sqrt(2)*SNR*FPR) as a toy model.
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    fpr = np.linspace(0, 1, 500)

    benchmarks = [
        (r"$m=10^{-14}$ eV, $g=10^{-12}$ eV$^{-1}$", 3.0,  "#2166ac"),
        (r"$m=10^{-14}$ eV, $g=10^{-13}$ eV$^{-1}$", 1.5,  "#4dac26"),
        (r"$m=10^{-14}$ eV, $g=10^{-14}$ eV$^{-1}$", 0.5,  "#d6604d"),
    ]

    for label, snr, color in benchmarks:
        from scipy.special import erfinv
        tpr = 0.5 * (1 + erf((np.sqrt(2) * erfinv(2*fpr - 1) + snr) / np.sqrt(2)))
        ax.plot(fpr, tpr, color=color, lw=2.0, label=f"{label}\n(SNR={snr})")

    ax.plot([0,1],[0,1], "k--", lw=0.8, label="Random classifier")
    ax.axvline(1e-3, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.text(1.2e-3, 0.1, r"$s_\mathrm{thresh}$", color="gray", fontsize=8)

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (detection efficiency)")
    ax.set_title("Figure 4 (right): ROC curve\n"
                 r"[PLACEHOLDER — replace with trained GNN output]",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    fig.tight_layout()
    return fig

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    out_dir = r"C:\Users\mitch\OneDrive\Pictures\iPod Photo Cache"
    os.makedirs(out_dir, exist_ok=True)

    print("Generating Figure 2: memory kernel + spectral density...")
    fig2 = make_memory_kernel_figure()
    fig2.savefig(f"{out_dir}/fig2_memory_kernel.pdf", bbox_inches="tight")
    fig2.savefig(f"{out_dir}/fig2_memory_kernel.png", bbox_inches="tight", dpi=200)
    print("  Saved fig2_memory_kernel.pdf/.png")

    print("Generating Figure 3: sensitivity curves...")
    fig3 = make_figure()
    fig3.savefig(f"{out_dir}/fig3_sensitivity_curves.pdf", bbox_inches="tight")
    fig3.savefig(f"{out_dir}/fig3_sensitivity_curves.png", bbox_inches="tight", dpi=200)
    print("  Saved fig3_sensitivity_curves.pdf/.png")

    print("Generating Figure 4 (ROC placeholder)...")
    fig4 = make_roc_placeholder()
    fig4.savefig(f"{out_dir}/fig4_roc_placeholder.pdf", bbox_inches="tight")
    fig4.savefig(f"{out_dir}/fig4_roc_placeholder.png", bbox_inches="tight", dpi=200)
    print("  Saved fig4_roc_placeholder.pdf/.png")

    print("Done.")
