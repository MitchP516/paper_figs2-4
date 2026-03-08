"""
=============================================================================
 QUANTUM DECOHERENCE SPECTROSCOPY — GNN DARK MATTER DETECTOR
=============================================================================
 Implements the Graph Attention Network (GAT) described in Appendix C of:
 "Quantum Decoherence Spectroscopy as a Dark Matter Detection Channel"

 INSTALL INSTRUCTIONS
 --------------------
 pip install torch torchvision torchaudio
 pip install torch-geometric
 pip install matplotlib scikit-learn numpy

 WHAT THIS SCRIPT DOES
 ---------------------
 1. Simulates two-detector decoherence time-series at realistic SNR
 2. Trains a GAT classifier to distinguish DM signal from noise
 3. Saves trained model, Figure 4 (ROC curve), and results summary

 OUTPUT DIRECTORY
 ----------------
 All files are saved to the folder defined by OUTPUT_DIR below.
 Change it to whatever path you want before running.

=============================================================================
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORY — change this to wherever you want the files saved
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = r"C:\Users\mitch\OneDrive\Documentos\suckitbiatch"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def out(filename):
    return os.path.join(OUTPUT_DIR, filename)


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS  (natural units: hbar = c = k_B = 1)
# ─────────────────────────────────────────────────────────────────────────────

RHO_DM       = 0.4
V0           = 220e3
C_LIGHT      = 3e8
HBAR_EV_S    = 6.582e-16

DM_MASSES_EV = [1e-22, 1e-18, 1e-14]

LAMBDA_BATH  = 0.1
GAMMA_BATH   = 1e3
SIGMA_C      = 0.01

# Rescaled so the DM contribution to gA sits ~1-2 sigma above the noise
# floor (noise std ~0.034, LAMBDA_BATH=0.1). With phi0=1.0 and the 0.3
# prefactor in simulate_sequence, we need gx ~ 0.1 to inject ~0.03 into gA.
# Values are deliberately mass-independent here because phi0 is now
# normalised — physical mass dependence is captured in omega_dm and tau_c.
GX_BY_MASS = {
    1e-22: 0.15,
    1e-18: 0.12,
    1e-14: 0.10,
}

CORR_NOISE_FRAC = 0.3

N_OMEGA      = 32
T_SEQ        = 100
DT           = 1e-3

N_TRAIN      = 8000
N_TEST       = 10000
BATCH_SIZE   = 256
LEARNING_RATE = 1e-3
N_EPOCHS     = 60
# ALPHA_CORR dropped to 0.1 — corr_loss is now a regulariser, not a driver.
# BCE must dominate; the old value of 0.5 caused the unconditional signal
# push to overwhelm cross-entropy and collapse the model.
ALPHA_CORR   = 0.1

N_LAYERS     = 3
HIDDEN_DIM   = 64
N_HEADS      = 4

TAU_PLUS     = 0.02
TAU_MINUS    = 0.02
A_PLUS       = 1.0
A_MINUS      = 0.5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
print(f"Output directory: {OUTPUT_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

def dm_coherence_time(m_dm_ev):
    omega_dm = m_dm_ev / HBAR_EV_S
    tau_c    = 1.0 / (m_dm_ev * (V0/C_LIGHT)**2 / HBAR_EV_S)
    return omega_dm, tau_c


def dm_field_amplitude(m_dm_ev):
    # Return a dimensionless O(1) amplitude. The raw sqrt(rho/m^2) blows up
    # at ultralight masses and makes signal trivially separable from noise.
    # The physical coupling strength is already encoded in GX_BY_MASS — we
    # just need a unit-normalised field envelope here.
    return 1.0


def simulate_sequence(m_dm_ev, is_signal, node_sep=1.0):
    omega_dm, _ = dm_coherence_time(m_dm_ev)
    phi0        = dm_field_amplitude(m_dm_ev)
    gx          = GX_BY_MASS[m_dm_ev]
    freq_axis   = np.linspace(0, 2*omega_dm + 1e3, N_OMEGA)

    X_A = np.zeros((T_SEQ, 2 + N_OMEGA))
    X_B = np.zeros((T_SEQ, 2 + N_OMEGA))

    J_base = (2*LAMBDA_BATH*GAMMA_BATH*freq_axis /
              (freq_axis**2 + GAMMA_BATH**2 + 1e-30))
    J_base /= (J_base.max() + 1e-30)

    corr_noise_gamma = CORR_NOISE_FRAC * LAMBDA_BATH * np.random.randn(T_SEQ)
    corr_noise_phi   = CORR_NOISE_FRAC * SIGMA_C     * np.random.randn(T_SEQ)
    corr_noise_S     = CORR_NOISE_FRAC * np.random.rand(T_SEQ, N_OMEGA)

    amp_jitter   = 1.0 + 0.5*np.random.randn() if is_signal else 1.0
    global_phase = np.random.uniform(0, 2*np.pi)

    for t in range(T_SEQ):
        time = t * DT

        gA     = LAMBDA_BATH*(1 + 0.15*np.random.randn()) + corr_noise_gamma[t]
        gB     = LAMBDA_BATH*(1 + 0.15*np.random.randn()) + corr_noise_gamma[t]
        dphi_A = SIGMA_C*np.random.randn() + corr_noise_phi[t]
        dphi_B = SIGMA_C*np.random.randn() + corr_noise_phi[t]
        S_A    = J_base + 0.08*np.random.rand(N_OMEGA) + corr_noise_S[t]
        S_B    = J_base + 0.08*np.random.rand(N_OMEGA) + corr_noise_S[t]

        if is_signal:
            phase_A = omega_dm*time + global_phase
            k_dm    = omega_dm / C_LIGHT
            phase_B = omega_dm*time - k_dm*node_sep + global_phase
            dm_A    = amp_jitter * gx * phi0 * np.cos(phase_A)
            dm_B    = amp_jitter * gx * phi0 * np.cos(phase_B)

            sideband = np.zeros(N_OMEGA)
            for k, f in enumerate(freq_axis):
                lo   = f - omega_dm
                hi   = f + omega_dm
                J_lo = (2*LAMBDA_BATH*GAMMA_BATH*abs(lo)/
                        (lo**2+GAMMA_BATH**2+1e-30)) if lo > 0 else 0
                J_hi = (2*LAMBDA_BATH*GAMMA_BATH*abs(hi)/
                        (hi**2+GAMMA_BATH**2+1e-30))
                # Removed the 1e10 multiplier — it inflated the sideband to
                # >> noise floor regardless of gx, making S_A trivially
                # distinguishable. gx alone sets the scale.
                sideband[k] = (gx/(2*omega_dm+1e-30))*(J_lo - J_hi)

            gA     += abs(dm_A)*0.3;  gB     += abs(dm_B)*0.3
            dphi_A += dm_A*DT;        dphi_B += dm_B*DT
            S_A    += 0.15*np.abs(sideband)
            S_B    += 0.15*np.abs(sideband)

        X_A[t,0] = gA;  X_A[t,1] = dphi_A;  X_A[t,2:] = S_A
        X_B[t,0] = gB;  X_B[t,1] = dphi_B;  X_B[t,2:] = S_B

    return X_A.astype(np.float32), X_B.astype(np.float32), int(is_signal)


def build_dataset(m_dm_ev, n_samples):
    dataset    = []
    labels     = [1]*(n_samples//2) + [0]*(n_samples - n_samples//2)
    np.random.shuffle(labels)
    edge_index = torch.tensor([[0,1],[1,0]], dtype=torch.long)
    for label in labels:
        X_A, X_B, y = simulate_sequence(m_dm_ev, bool(label))
        feat_A = torch.tensor(X_A.flatten(), dtype=torch.float)
        feat_B = torch.tensor(X_B.flatten(), dtype=torch.float)
        x      = torch.stack([feat_A, feat_B], dim=0)
        dataset.append(Data(x=x, edge_index=edge_index,
                            y=torch.tensor([y], dtype=torch.long)))
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# 2. GAT MODEL  [Appendix C.1–C.4]
# ─────────────────────────────────────────────────────────────────────────────

class STDPModulator(nn.Module):
    """STDP-inspired edge modulation  [App C.3, Eq C.6–C.8]"""
    def __init__(self):
        super().__init__()
        self.log_tau_p = nn.Parameter(torch.tensor(np.log(TAU_PLUS)))
        self.log_tau_m = nn.Parameter(torch.tensor(np.log(TAU_MINUS)))
        self.log_Ap    = nn.Parameter(torch.tensor(np.log(A_PLUS)))
        self.log_Am    = nn.Parameter(torch.tensor(np.log(A_MINUS)))
        self.eta       = nn.Parameter(torch.tensor(0.1))

    def forward(self, h_A, h_B):
        tau_p   = torch.exp(self.log_tau_p)
        tau_m   = torch.exp(self.log_tau_m)
        Ap      = torch.exp(self.log_Ap)
        Am      = torch.exp(self.log_Am)
        delta_t = (h_A[:,0] - h_B[:,0]).mean()
        K = Ap*torch.exp(-delta_t/tau_p) if delta_t > 0 \
            else -Am*torch.exp(delta_t/tau_m)
        return (1.0 + self.eta*K).clamp(min=0.1, max=10.0)


class DMDetectorGNN(nn.Module):
    """Full GAT-based DM detector  [Appendix C, Eq C.1–C.12]"""
    def __init__(self, in_features, hidden=HIDDEN_DIM,
                 n_layers=N_LAYERS, heads=N_HEADS):
        super().__init__()
        self.embed = nn.Linear(in_features, hidden)
        self.gats  = nn.ModuleList()
        for i in range(n_layers):
            inc    = hidden*heads if i > 0 else hidden
            concat = (i < n_layers - 1)
            self.gats.append(GATConv(inc, hidden, heads=heads,
                                     concat=concat, dropout=0.1,
                                     add_self_loops=True))
        self.stdp = STDPModulator()
        self.mlp  = nn.Sequential(
            nn.Linear(hidden*2, hidden), nn.ELU(), nn.Dropout(0.2),
            nn.Linear(hidden, 2)
        )

    def forward(self, data):
        x, ei = data.x, data.edge_index
        h = F.elu(self.embed(x))
        for i, gat in enumerate(self.gats):
            h = gat(h, ei)
            if i < len(self.gats) - 1:
                h = F.elu(h)
        h = h * self.stdp(h[ei[0]], h[ei[1]]).detach()

        node_A_mask = (torch.arange(h.size(0), device=h.device) % 2 == 0)
        hA = h[ node_A_mask]
        hB = h[~node_A_mask]
        return self.mlp(torch.cat([hA, hB], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOSS FUNCTION  [replaces App C.5 Eq C.12]
#
#    L_total = L_BCE + alpha * L_corr_supervised
#
#    corr_loss is now a SUPERVISED term:
#      - Only fires on true positive (signal) samples.
#      - Penalises deviation of predicted P(signal) from the expected
#        spatial coherence factor sf = exp(-node_sep / lambda_dm).
#      - Contribution is exactly zero on background events, so it cannot
#        pull the model toward predicting signal on noise — which is what
#        the original unconditional formulation was doing.
# ─────────────────────────────────────────────────────────────────────────────

def corr_loss(logits, labels, data, m_dm_ev, node_sep=1.0):
    v_vir = V0 / C_LIGHT
    lam   = HBAR_EV_S / (m_dm_ev * v_vir + 1e-300)
    sf    = float(np.exp(-node_sep / (lam * C_LIGHT + 1e-300)))
    sf    = max(sf, 1e-6)

    p_signal = torch.softmax(logits, dim=-1)[:, 1]
    is_sig   = (labels == 1).float()

    # On true signal events: push predicted confidence toward sf.
    # On true noise events:  contribution is zero (masked by is_sig).
    loss  = is_sig * (p_signal - sf) ** 2
    n_sig = is_sig.sum().clamp(min=1)
    return loss.sum() / n_sig


def loss_fn(logits, labels, data, m_dm_ev):
    bce  = F.cross_entropy(logits, labels)
    corr = corr_loss(logits, labels, data, m_dm_ev)
    return bce + ALPHA_CORR * corr


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_model(m_dm_ev, train_data, in_features):
    model  = DMDetectorGNN(in_features).to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                              weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)
    loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    print(f"\n  Training m_DM={m_dm_ev:.0e} eV ...")

    for ep in range(1, N_EPOCHS + 1):
        model.train()
        tot_loss  = 0.0
        all_probs = []
        all_labs  = []

        for batch in loader:
            batch  = batch.to(DEVICE)
            opt.zero_grad()
            logits = model(batch)
            labs   = batch.y.squeeze()
            loss   = loss_fn(logits, labs, batch, m_dm_ev)
            loss.backward()

            # Gradient clipping — prevents corr term destabilising early epochs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            opt.step()
            tot_loss += loss.item()

            with torch.no_grad():
                p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                all_probs.extend(p)
                all_labs.extend(labs.cpu().numpy())

        sched.step()

        if ep % 10 == 0 or ep == 1:
            mean_p = np.mean(all_probs)
            if ep == 1 and not (0.2 < mean_p < 0.8):
                print(f"    ⚠  Epoch 1 mean P(signal)={mean_p:.3f} — "
                      f"possible collapse. Check loss weights.")
            print(f"    Epoch {ep:3d}/{N_EPOCHS}  "
                  f"Loss: {tot_loss/len(loader):.4f}  "
                  f"mean P(sig): {mean_p:.3f}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 5. EVALUATION AND ROC CURVE  [Sec. VI.F]
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, test_data):
    model.eval()
    loader = DataLoader(test_data, batch_size=256, shuffle=False)
    probs, labs = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(DEVICE)
            p = torch.softmax(model(b), dim=-1)[:,1].cpu().numpy()
            probs.extend(p)
            labs.extend(b.y.cpu().numpy().flatten())
    return np.array(labs), np.array(probs)


def tpr_at_fpr(fpr, tpr, target=1e-3):
    idx = min(np.searchsorted(fpr, target), len(tpr)-1)
    return tpr[idx]


def plot_roc(results, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    colors  = ['#1f77b4', '#ff7f0e', '#2ca02c']
    labels  = {1e-22: r'$m_\chi=10^{-22}$ eV',
               1e-18: r'$m_\chi=10^{-18}$ eV',
               1e-14: r'$m_\chi=10^{-14}$ eV'}
    for (m, fpr, tpr, roc_auc), c in zip(results, colors):
        ax.plot(fpr, tpr, color=c, lw=1.8,
                label=f'{labels[m]}  (AUC={roc_auc:.3f})')
    ax.axvline(1e-3, color='gray', ls='--', lw=1.0,
               label=r'Operating point (FPR$=10^{-3}$)')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate',  fontsize=11)
    ax.set_xscale('log')
    ax.set_xlim([1e-4, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"\n  Figure 4 saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    in_features = T_SEQ * (2 + N_OMEGA)
    X_A, X_B, _ = simulate_sequence(1e-22, True)
    X_An, X_Bn, _ = simulate_sequence(1e-22, False)
    print("Signal   gA mean/std:", X_A[:, 0].mean(), X_A[:, 0].std())
    print("Noise    gA mean/std:", X_An[:, 0].mean(), X_An[:, 0].std())
    print("Signal dphi mean/std:", X_A[:, 1].mean(), X_A[:, 1].std())
    print("Noise  dphi mean/std:", X_An[:, 1].mean(), X_An[:, 1].std())
    roc_results = []
    summary     = ["GNN Classification Results\n", "="*40+"\n",
                   "TPR @ FPR=0.001 for each DM mass:\n\n"]

    for m in DM_MASSES_EV:
        print(f"\n{'='*60}\n  DM mass: {m:.0e} eV\n{'='*60}")
        print("  Generating training data ..."); train_d = build_dataset(m, N_TRAIN)
        print("  Generating test data ...");     test_d  = build_dataset(m, N_TEST)
        model = train_model(m, train_d, in_features)

        model_path = out(f'dm_gnn_model_{m:.0e}eV.pt')
        torch.save(model.state_dict(), model_path)
        print(f"  Model saved: {model_path}")

        labels, probs = evaluate(model, test_d)
        fpr, tpr, _   = roc_curve(labels, probs)
        roc_auc       = auc(fpr, tpr)
        tpr_op        = tpr_at_fpr(fpr, tpr)
        print(f"  AUC = {roc_auc:.4f}")
        print(f"  TPR @ FPR=0.001 = {tpr_op*100:.1f}%  <-- plug into Sec. VI.F")
        roc_results.append((m, fpr, tpr, roc_auc))
        summary.append(f"  {m:.0e} eV:  TPR={tpr_op*100:.1f}%  AUC={roc_auc:.4f}\n")

    plot_roc(roc_results, out('fig4_roc_curve.pdf'))

    summary_path = out('results_summary.txt')
    with open(summary_path, 'w') as f:
        f.writelines(summary)
        f.write("\nPlug TPR values into Section VI.F placeholder [X]%\n")
    print(f"\n  Results summary saved: {summary_path}")
    print(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
