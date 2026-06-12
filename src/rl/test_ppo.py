import torch
import sys
import numpy as np
from pathlib import Path
import random

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
from scipy import stats
# %%  ── Config ──────────────────────────────────────────────────────────────
DATASET   = "dow30"           # "dow30" | "ndx100" | "sse50"
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

sharpes = []

# for SEED in range(100):

#     random.seed(SEED)
#     np.random.seed(SEED)

#     torch.manual_seed(SEED)

#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(SEED)

#     test_rewards = []

#     with torch.no_grad():

#         for t in range(len(X_test)):

#             weights, _, _, _ = model(
#                 X_test[t].to(device),
#                 g_test[t]["edge_index"].to(device),
#                 static_graph["edge_index"].to(device),
#             )

#             reward = env.step(
#                 weights,
#                 y_test[t]
#             )

#             test_rewards.append(
#                 reward.item()
#             )

#     sharpe = compute_sharpe(
#         test_rewards
#     )

#     sharpes.append(sharpe)

# sharpes = np.array(sharpes)

# best_seed = np.argmax(sharpes)
# print("\n================ EQUAL WEIGHT ================")

# equal_rewards = []

# N = y_test.shape[1]

# equal_weights = torch.ones(
#     N,
#     device=device
# ) / N

# for t in range(len(y_test)):

#     reward = torch.sum(
#         equal_weights *
#         y_test[t].to(device)
#     )

#     equal_rewards.append(
#         reward.item()
#     )

# equal_sharpe = compute_sharpe(
#     equal_rewards
# )

# equal_annual_return = (
#     np.prod(
#         1 + np.array(equal_rewards)
#     )
# ) ** (
#     252 / len(equal_rewards)
# ) - 1

# equal_volatility = (
#     np.std(equal_rewards)
#     * np.sqrt(252)
# )

# equal_max_dd = compute_max_drawdown(
#     equal_rewards
# )

# print(
#     f"Sharpe Ratio : "
#     f"{equal_sharpe:.4f}"
# )

# print(
#     f"Annual Return: "
#     f"{equal_annual_return:.4f}"
# )

# print(
#     f"Volatility   : "
#     f"{equal_volatility:.4f}"
# )

# print(
#     f"Max Drawdown : "
#     f"{equal_max_dd:.4f}"
# )


# print("\n================ SHARPE DISTRIBUTION ================")

# print(
#     f"Mean Sharpe : {sharpes.mean():.4f}"
# )
# print("Median Sharpe:", np.median(sharpes))
# # print("Mode Sharpe:", np.mode(sharpes))
# print(
#     "Mode Sharpe:",
#     stats.mode(
#         np.round(sharpes, 2),
#         keepdims=False
#     )
# )

# print(
#     f"Std Sharpe  : {sharpes.std():.4f}"
# )

# print(
#     f"Min Sharpe  : {sharpes.min():.4f}"
# )

# print(
#     f"Max Sharpe  : {sharpes.max():.4f}"
# )

# print(
#     f"Best Seed   : {best_seed}"
# )

# print(
#     f"Best Sharpe : {sharpes[best_seed]:.4f}"
# )

# top5 = np.argsort(sharpes)[::-1][:5]

# print("\nTop 5 Seeds")

# for idx in top5:

#     print(
#         f"Seed {idx:03d} -> "
#         f"{sharpes[idx]:.4f}"
#     )



# print("\n================ COMPARISON =================")

# print(
#     f"DGDRL Mean Sharpe : "
#     f"{sharpes.mean():.4f}"
# )

# print(
#     f"DGDRL Best Sharpe : "
#     f"{sharpes.max():.4f}"
# )

# print(
#     f"Equal Weight      : "
#     f"{equal_sharpe:.4f}"
# )
# print(
#     "Pct > Equal Weight:",
#     np.mean(sharpes > equal_sharpe) * 100
# )

with torch.no_grad():

    _, _, _, E_final = model(
        X_test[0],
        g_test[0]["edge_index"],
        static_graph["edge_index"]
    )

    scores = actor.rmlp(E_final)

    print("score mean:", scores.mean().item())
    print("score std :", scores.std().item())
    print("score min :", scores.min().item())
    print("score max :", scores.max().item())

with torch.no_grad():

    _, _, _, E_final = model(
        X_test[0],
        g_test[0]["edge_index"],
        static_graph["edge_index"]
    )

    scores = actor.rmlp(E_final).squeeze(-1)

    alpha = torch.nn.functional.softplus(scores) + 1.0

    print("alpha mean:", alpha.mean().item())
    print("alpha std :", alpha.std().item())
    print("alpha min :", alpha.min().item())
    print("alpha max :", alpha.max().item())

with torch.no_grad():

    weights, _, _, _ = model(
        X_test[0],
        g_test[0]["edge_index"],
        static_graph["edge_index"]
    )

    print("weight std :", weights.std())
    print("weight min :", weights.min())
    print("weight max :", weights.max())


with torch.no_grad():

    _, _, _, E_final = model(
        X_test[0],
        g_test[0]["edge_index"],
        static_graph["edge_index"]
    )

    scores = actor.rmlp(E_final).squeeze(-1)

    alpha = torch.nn.functional.softplus(scores) + 1.0

    mean_weights = alpha / alpha.sum()

    print("mean weight std :", mean_weights.std())
    print("mean weight min :", mean_weights.min())
    print("mean weight max :", mean_weights.max())

scores = actor.rmlp(E_final).squeeze(-1)

print(torch.argsort(scores, descending=True)[:10])