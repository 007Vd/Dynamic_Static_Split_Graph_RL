# train_ppo.py — Training loop per DGDRL Algorithm 1
#
# Stopping strategy: early stopping on validation Sharpe ratio.
# No fixed epoch count is given in the paper — it runs until convergence.
# We use a held-out validation slice (last VAL_FRAC of training data)
# and stop when validation Sharpe has not improved for PATIENCE epochs.
# Expected convergence: 50–150 epochs depending on dataset size.

# %%
import torch
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.models.lstm_ha import LSTMHA
from src.models.mgan import DynamicMGAN, StaticMGAN
from src.models.fusion import FusionLayer
from src.rl.actor import Actor
from src.rl.critic import Critic
from src.models.dgdrl import DGDRL
from src.rl.ppo import PPO
from src.rl.buffer import RolloutBuffer
from src.rl.portfolio_env import PortfolioEnv

# %%  ── Config ──────────────────────────────────────────────────────────────
DATASET   = "ndx100"           # "dow30" | "ndx100" | "sse50"
MAX_EPOCHS = 50           # hard ceiling — should never be reached
PATIENCE   = 20         # stop if val Sharpe doesn't improve for this many epochs
VAL_FRAC   = 0.1              # fraction of training data held out for validation
SAVE_PATH  = PROJECT_ROOT / f"checkpoints/{DATASET}_best.pt"
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

# %%  ── Load data ──────────────────────────────────────────────────────────
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
torch.backends.cudnn.benchmark = True

print("Using device:", device)
DATA_ROOT = PROJECT_ROOT / "data"

X_aligned              = torch.load(DATA_ROOT / f"aligned/{DATASET}/X_aligned.pt")
y_aligned              = torch.load(DATA_ROOT / f"aligned/{DATASET}/y_aligned.pt")
dynamic_graphs_aligned = torch.load(DATA_ROOT / f"aligned/{DATASET}/dynamic_graphs_aligned.pt")
static_graph           = torch.load(DATA_ROOT / f"graphs/{DATASET}/static_graph.pt")

print("X shape     :", X_aligned.shape)
print("y shape     :", y_aligned.shape)
print("# dyn graphs:", len(dynamic_graphs_aligned))

assert len(X_aligned) == len(y_aligned)
assert len(X_aligned) == len(dynamic_graphs_aligned)
print(X_aligned.shape)
# %%  ── Train / val split ──────────────────────────────────────────────────
# Keep chronological order — val is the LAST VAL_FRAC of the training window.
T         = len(X_aligned)
train_end=int(T*0.8)
val_end=int(T*0.9)


X_train   = X_aligned[:train_end]
X_train=X_train.to(device)
y_train   = y_aligned[:train_end]
y_train=y_train.to(device)
g_train = dynamic_graphs_aligned[:train_end]


X_val     = X_aligned[train_end:val_end]
y_val     = y_aligned[train_end:val_end]
g_val     = dynamic_graphs_aligned[train_end:val_end]

X_test   = X_aligned[val_end:]
y_test   = y_aligned[val_end:]
g_test   = dynamic_graphs_aligned[val_end:]

print(f"Train steps : {len(X_train)}  |  Val steps : {len(X_val)}")

# %%  ── Sharpe helper ───────────────────────────────────────────────────────
def compute_sharpe(rewards: list, annualisation: float = 252.0) -> float:
    """Annualised Sharpe ratio from a list of daily portfolio returns."""
    r = np.array(rewards, dtype=np.float64)
    if r.std() < 1e-9:
        return 0.0
    return float((r.mean() / r.std()) * np.sqrt(annualisation))

def compute_max_drawdown(returns):

    returns = np.array(returns)
    equity_curve = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(equity_curve)

    drawdowns = (equity_curve -running_max) / running_max

    return drawdowns.min()
# %%  ── Build model ─────────────────────────────────────────────────────────
lstm_ha      = LSTMHA()
dynamic_mgan = DynamicMGAN()
static_mgan  = StaticMGAN()
fusion       = FusionLayer()
actor        = Actor()
critic       = Critic()

model = DGDRL(
    lstm_ha=lstm_ha,
    dynamic_mgan=dynamic_mgan,
    static_mgan=static_mgan,
    fusion=fusion,
    actor=actor,
    critic=critic,
)
model=model.to(device)
print(
    "Model device:",
    next(model.parameters()).device
)
ppo    = ppo = PPO(
    model=model
)
env    = PortfolioEnv()
buffer = RolloutBuffer()


# %%  ── Training loop ───────────────────────────────────────────────────────
best_val_sharpe  = -np.inf
epochs_no_improve = 0

for epoch in range(1, MAX_EPOCHS + 1):

    # ── Collect trajectory (Algorithm 1 Step 2) ──────────────────────────
    model.train()
    buffer.clear()
    train_rewards = []

    for t in range(len(X_train)):
        X_t     = X_train[t]
        graph_t = g_train[t]
        y_t     = y_train[t]

        weights, log_prob, value, E_final = model(
    X_t,
    graph_t["edge_index"].to(device),
    static_graph["edge_index"].to(device),
)

        reward = env.step(weights, y_t)

        # buffer.states.append(E_final.detach())
        buffer.X.append(X_t)

        buffer.dynamic_edges.append(
    graph_t["edge_index"].to(device)
)

        buffer.static_edges.append(
    static_graph["edge_index"].to(device)
)
        buffer.actions.append(weights.detach())
        buffer.values.append(value.detach())
        buffer.log_probs.append(log_prob.detach())
        buffer.rewards.append(reward.detach())
        train_rewards.append(reward.item())

    # ── PPO update (Algorithm 1 Steps 5-10) ──────────────────────────────
    ppo.update(buffer)

    train_sharpe = compute_sharpe(train_rewards)

    # ── Validation pass (no grad, no buffer update) ───────────────────────
    model.eval()
    val_rewards = []

    with torch.no_grad():
        for t in range(len(X_val)):
            weights, _, _, _ = model(
                X_val[t],
                g_val[t]["edge_index"],
                static_graph["edge_index"],
            )
            reward = env.step(weights, y_val[t])
            val_rewards.append(reward.item())

    val_sharpe = compute_sharpe(val_rewards)

    print(
        f"Epoch {epoch:03d} | "
        f"Train Sharpe = {train_sharpe:+.4f} | "
        f"Val Sharpe = {val_sharpe:+.4f}"
        + (" ✓ best" if val_sharpe > best_val_sharpe else "")
    )

    # ── Checkpoint if improved ────────────────────────────────────────────
    if val_sharpe > best_val_sharpe:
        best_val_sharpe   = val_sharpe
        epochs_no_improve = 0
        torch.save({
            "epoch":          epoch,
            "model_state":    model.state_dict(),
            "val_sharpe":     val_sharpe,
            "train_sharpe":   train_sharpe,
        }, SAVE_PATH)
    else:
        epochs_no_improve += 1

    # ── Early stopping ────────────────────────────────────────────────────
    if epochs_no_improve >= PATIENCE:
        print(
            f"\nEarly stopping at epoch {epoch} — "
            f"val Sharpe did not improve for {PATIENCE} consecutive epochs."
        )
        print(f"Best val Sharpe: {best_val_sharpe:+.4f}  →  checkpoint: {SAVE_PATH}")
        break

else:
    print(f"\nReached MAX_EPOCHS ({MAX_EPOCHS}). Best val Sharpe: {best_val_sharpe:+.4f}")
 # %%
# testing
def compute_sharpe(rewards, annualisation=252):
    r = np.array(rewards, dtype=np.float64)

    if r.std() < 1e-9:
        return 0.0

    return (r.mean() / r.std()) * np.sqrt(annualisation)


def compute_max_drawdown(returns):

    equity_curve = np.cumprod(
        1 + np.array(returns)
    )

    running_max = np.maximum.accumulate(
        equity_curve
    )

    drawdowns = (
        equity_curve -
        running_max
    ) / running_max

    return drawdowns.min()

checkpoint = torch.load(
    PROJECT_ROOT / f"checkpoints/{DATASET}_best.pt",
    map_location="cpu"
)

model.load_state_dict(
    checkpoint["model_state"]
)

print(
    f"Loaded epoch {checkpoint['epoch']} "
    f"| Val Sharpe = {checkpoint['val_sharpe']:.4f}"
)

model.eval()

print("Test samples:", len(X_test))

test_rewards = []
all_weights = []

with torch.no_grad():

    for t in range(len(X_test)):

        weights, _, _, _ = model(
    X_test[t].to(device),
    g_test[t]["edge_index"].to(device),
    static_graph["edge_index"].to(device),
)

        reward = env.step(
            weights,
            y_test[t]
        )

        test_rewards.append(
            reward.item()
        )

        all_weights.append(
            weights.cpu().numpy()
        )

test_sharpe = compute_sharpe(
    test_rewards
)

annual_return = (
    np.prod(
        1 + np.array(test_rewards)
    )
) ** (
    252 / len(test_rewards)
) - 1

volatility = (
    np.std(test_rewards)
    * np.sqrt(252)
)

max_dd = compute_max_drawdown(
    test_rewards
)

all_weights = np.array(all_weights)

mean_weights = all_weights.mean(axis=0)

weight_std = mean_weights.std()

weight_min = mean_weights.min()

weight_max = mean_weights.max()

# %%
equal_rewards = []

N = y_test.shape[1]

equal_weights = torch.ones(N) / N

for t in range(len(y_test)):

    reward = torch.sum(
        equal_weights * y_test[t]
    )

    equal_rewards.append(
        reward.item()
    )

equal_sharpe = compute_sharpe(
    equal_rewards
)

equal_annual_return = (
    np.prod(
        1 + np.array(equal_rewards)
    )
) ** (
    252 / len(equal_rewards)
) - 1

equal_volatility = (
    np.std(equal_rewards)
    * np.sqrt(252)
)

equal_max_dd = compute_max_drawdown(
    equal_rewards
)

#%%
print("\n================ DGDRL TEST =================")

print(f"Sharpe Ratio : {test_sharpe:.4f}")
print(f"Annual Return: {annual_return:.4f}")
print(f"Volatility   : {volatility:.4f}")
print(f"Max Drawdown : {max_dd:.4f}")

print("\n================ EQUAL WEIGHT ===============")

print(f"Sharpe Ratio : {equal_sharpe:.4f}")
print(f"Annual Return: {equal_annual_return:.4f}")
print(f"Volatility   : {equal_volatility:.4f}")
print(f"Max Drawdown : {equal_max_dd:.4f}")

print("\n================ WEIGHT ANALYSIS ============")

print(f"Mean Weight Min : {weight_min:.6f}")
print(f"Mean Weight Max : {weight_max:.6f}")
print(f"Mean Weight Std : {weight_std:.6f}")
# print(f"HHI             : {hhi:.6f}")

print("\nTop 10 Average Allocations")

top_idx = np.argsort(mean_weights)[::-1][:10]

for rank, idx in enumerate(top_idx, start=1):

    print(
        f"{rank:02d}. "
        f"Stock {idx:02d} "
        f"-> {mean_weights[idx]:.4f}"
    )
# %%
with torch.no_grad():

    weights, _, _, E_final = model(
        X_test[0],
        g_test[0]["edge_index"],
        static_graph["edge_index"]
    )

    scores = actor.rmlp(E_final)

    print("score std:", scores.std())
    print("score min:", scores.min())
    print("score max:", scores.max())
# %%
