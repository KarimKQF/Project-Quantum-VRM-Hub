import os
import sys
import glob
import time
import random
import pandas as pd
import numpy as np
import torch
from datetime import datetime

# ============================================================
# GA SHARPE HUNTER - AI POWERED
# ------------------------------------------------------------
# 1. DATA   : Reads 'DATA_LAKE' (Offline) + Flexible filtering
# 2. TURBO  : Generates 50,000 portfolios at startup
# 3. GPU    : RTX 4070 (PyTorch CUDA) + Anti-Crash Batch Size
# 4. HOF    : Full history saved in 'HALL_OF_FAME.csv'
# ============================================================

# --- CPU OPTIMIZATION ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# --- GPU CONFIGURATION ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    try:
        torch.set_float32_matmul_precision('medium')
    except Exception:
        pass
    print(f"🚀 GPU DETECTED: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ GPU NOT DETECTED. Swithing to CPU mode.")

# --- PARAMETERS ---
DATA_LAKE_DIR = "datalake-compact"
CHAMPION_FILE = "TOP_1_GLOBAL.csv"
HISTORY_FILE = "HALL_OF_FAME.csv"

# Number of lake files loaded per cycle (50 * 100 approx. 5000 assets)
FILES_PER_CYCLE = 50

N_CYCLES = 100000

# List of known high-performers to inject if found
# (Indian, European, and OTC stocks mixed)
KNOWN_MONSTERS = [
    "AUTOINT.BO", "MONT.PA", "NARMP.BO", "0JXF.L", "KEL.DE", "ALC.MC", "0SL.F", 
    "YUNSA.IS", "BLUESTARCO.NS", "IMPC.JK", "AEPLF", "6896.HK", "S7MB.BE", 
    "SHTRF", "GNG.AX", "CEBR3.SA", "BV4.BE", "GALLANTT.NS", "CTW.SG", "MDIKF", 
    "CRBJF", "SLNLF", "HAA1.MU", "KNG.V", "RBTK"
]

# GENETIC ALGORITHM PARAMETERS
ELITE_SIZE = 2000
POOL_SIZE = 25
POP_SIZE = 4096
N_GENERATIONS = 200
ELITISM = 500
CROSSOVER_RATE = 0.80
MUTATION_RATE = 0.80
MUTATION_K = 8
MIN_WEIGHT = 0.001
MAX_WEIGHT = 0.35
TARGET_SHARPE = 20.0
LAMBDA_CORR = 0.25
LEARNING_RATE = 0.08
OPTIM_STEPS = 100
FIXED_RF = 0.0315

# --- TURBO SETTINGS ---
TURBO_START_SIZE = 50000  # 50k portfolios generated for initial sorting

# Global Variables
T_MEAN = None
T_COV = None
T_RF = None
TICKER_TO_IDX = {}
CURRENT_ELITE = []

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# -----------------------------
# 1. LOCAL LOADING (DATA LAKE)
# -----------------------------
def get_local_data_cycle(cycle_id):
    """Loads a random subset of files from the data lake."""
    all_files = glob.glob(os.path.join(DATA_LAKE_DIR, "*.parquet"))

    if not all_files:
        log("❌ DATA_LAKE directory is empty or not found.")
        return pd.DataFrame(), []

    # Pick random files
    selected_files = random.sample(all_files, min(len(all_files), FILES_PER_CYCLE))

    dfs = []
    for f in selected_files:
        try:
            df = pd.read_parquet(f)
            # Soft cleaning: drop only if mostly empty
            df = df.dropna(axis=1, thresh=int(df.shape[0] * 0.6))
            if not df.empty:
                dfs.append(df)
        except Exception:
            pass

    if not dfs:
        return pd.DataFrame(), []

    # Merge and fill
    prices = pd.concat(dfs, axis=1)
    prices = prices.ffill().bfill()

    return prices, prices.columns.tolist()

# -----------------------------
# 2. GPU ENGINE (ANTI-CRASH)
# -----------------------------
def optimize_population_gpu(population_pools, w_init_list=None):
    """Optimizes weights for a batch of portfolios using PyTorch GPU."""
    pop_size = len(population_pools)
    # Target matrix indices
    indices_cpu = np.zeros((pop_size, POOL_SIZE), dtype=np.int64)

    # Fallback list to fill gaps
    valid_values = list(TICKER_TO_IDX.values())
    if not valid_values:
        valid_values = [0]

    for i, pool in enumerate(population_pools):
        idxs = [TICKER_TO_IDX.get(t, 0) for t in pool]

        # FIX CRASH: Auto-fill if not enough valid tickers
        while len(idxs) < POOL_SIZE:
            idxs.append(random.choice(valid_values))
        if len(idxs) > POOL_SIZE:
            idxs = idxs[:POOL_SIZE]

        indices_cpu[i] = idxs

    batch_indices = torch.tensor(indices_cpu, device=DEVICE)
    batch_means = T_MEAN[batch_indices]
    batch_cov = torch.zeros((pop_size, POOL_SIZE, POOL_SIZE), device=DEVICE)

    for i in range(pop_size):
        idx = batch_indices[i]
        batch_cov[i] = T_COV[idx][:, idx]

    if w_init_list is None:
        weights_logits = torch.randn((pop_size, POOL_SIZE), device=DEVICE, requires_grad=True)
    else:
        init_tensor = torch.randn((pop_size, POOL_SIZE), device=DEVICE)
        for i, w in enumerate(w_init_list):
            if w is not None and len(w) == POOL_SIZE:
                init_tensor[i] = torch.tensor(w, device=DEVICE)
        weights_logits = torch.log(torch.abs(init_tensor) + 1e-6).clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([weights_logits], lr=LEARNING_RATE)

    for _ in range(OPTIM_STEPS):
        optimizer.zero_grad()
        weights = torch.softmax(weights_logits, dim=1)
        port_returns = (weights * batch_means).sum(dim=1)
        port_vars = torch.einsum('bi,bij,bj->b', weights, batch_cov, weights)
        sharpes = (port_returns - T_RF) / torch.sqrt(port_vars + 1e-9)

        stds = torch.sqrt(torch.diagonal(batch_cov, dim1=1, dim2=2))
        outer = stds.unsqueeze(2) @ stds.unsqueeze(1)
        raw_corr = torch.einsum('bi,bij,bj->b', weights, batch_cov / (outer + 1e-9), weights)

        # Loss function: Maximize Sharpe, Minimize Correlation, Constraint Max Weight
        loss = -sharpes + (LAMBDA_CORR * (raw_corr - (weights**2).sum(dim=1))) + (torch.nn.functional.relu(weights - MAX_WEIGHT).sum(dim=1) * 20.0)
        loss.mean().backward()
        optimizer.step()

    final_weights = torch.clamp(torch.softmax(weights_logits, dim=1).detach(), MIN_WEIGHT, MAX_WEIGHT)
    final_weights /= final_weights.sum(dim=1, keepdim=True)
    f_ret = (final_weights * batch_means).sum(dim=1)
    f_vol = torch.sqrt(torch.einsum('bi,bij,bj->b', final_weights, batch_cov, final_weights) + 1e-9)
    
    return ((f_ret - T_RF) / f_vol).cpu().numpy(), final_weights.cpu().numpy()

# -----------------------------
# 3. GA OPERATORS
# -----------------------------
def random_pool():
    if not CURRENT_ELITE:
        return []
    s = min(len(CURRENT_ELITE), POOL_SIZE)
    p = random.sample(CURRENT_ELITE, s)
    while len(p) < POOL_SIZE:
        p.append(random.choice(CURRENT_ELITE))
    return p

def get_benchmark_pool():
    if not CURRENT_ELITE:
        return []
    valid_monster = [t for t in KNOWN_MONSTERS if t in TICKER_TO_IDX]
    if len(valid_monster) < POOL_SIZE:
        needed = POOL_SIZE - len(valid_monster)
        fill = [x for x in CURRENT_ELITE if x not in valid_monster]
        if fill:
            valid_monster += random.sample(fill, min(needed, len(fill)))
    while len(valid_monster) < POOL_SIZE:
        valid_monster.append(random.choice(CURRENT_ELITE))
    return valid_monster[:POOL_SIZE]

def mutate_pool(p):
    p = list(p)
    k_adj = min(len(p), MUTATION_K)
    out = random.sample(p, k_adj)
    cand = [x for x in CURRENT_ELITE if x not in p]
    if not cand:
        return p
    res = [x for x in p if x not in out] + random.sample(cand, min(len(cand), k_adj))
    while len(res) < POOL_SIZE:
        res.append(random.choice(CURRENT_ELITE))
    return res[:POOL_SIZE]

def crossover_pool(p1, p2):
    s1, s2 = set(p1), set(p2)
    comm = list(s1 & s2)
    rest = list(s1 ^ s2)
    random.shuffle(rest)
    child = comm + rest
    if len(child) >= POOL_SIZE:
        return child[:POOL_SIZE]
    needed = POOL_SIZE - len(child)
    cand = [x for x in CURRENT_ELITE if x not in child]
    if len(cand) < needed:
        return child + cand
    return child + random.sample(cand, needed)

# -----------------------------
# 4. CYCLE RUNNER
# -----------------------------
def run_cycle(cycle_id):
    global T_MEAN, T_COV, T_RF, TICKER_TO_IDX, CURRENT_ELITE

    PRICES, VALID_TICKERS = get_local_data_cycle(cycle_id)

    if PRICES.empty or len(VALID_TICKERS) < POOL_SIZE:
        log(f"⚠️ Cycle {cycle_id} : Not enough assets ({len(VALID_TICKERS)}). Skipping.")
        return None, None, -99.0

    # PREP GPU
    with np.errstate(invalid='ignore', divide='ignore'):
        log_rets = np.log(PRICES / PRICES.shift(1)).dropna()

    if log_rets.empty:
        return None, None, -99.0

    mean_ret = log_rets.mean().values * 252
    cov_ret = log_rets.cov().values * 252
    T_MEAN = torch.tensor(mean_ret, dtype=torch.float32, device=DEVICE)
    T_COV = torch.tensor(cov_ret, dtype=torch.float32, device=DEVICE)
    T_RF = torch.tensor(FIXED_RF, dtype=torch.float32, device=DEVICE)
    TICKER_TO_IDX = {t: i for i, t in enumerate(VALID_TICKERS)}

    # Elite (Soft Filter)
    vols = np.sqrt(np.diag(cov_ret))
    with np.errstate(invalid='ignore', divide='ignore'):
        sharpes = (mean_ret - FIXED_RF) / (vols + 1e-9)
    sharpes[~np.isfinite(sharpes)] = -99.0

    elite_idx = np.argsort(sharpes)[::-1][:ELITE_SIZE]
    CURRENT_ELITE = [VALID_TICKERS[i] for i in elite_idx]

    # Inject Monster if present in loaded data
    for m in KNOWN_MONSTERS:
        if m in VALID_TICKERS and m not in CURRENT_ELITE:
            CURRENT_ELITE.append(m)

    log(f"⚔️  Elite Pool: {len(CURRENT_ELITE)} assets")

    # --- TURBO START ---
    pre_pools = [random_pool() for _ in range(TURBO_START_SIZE)]
    if len(get_benchmark_pool()) > 0:
        pre_pools[0] = get_benchmark_pool()

    s_pre, w_pre = optimize_population_gpu(pre_pools, None)
    pre_evals = sorted([(float(s_pre[i]), pre_pools[i], w_pre[i]) for i in range(TURBO_START_SIZE)], key=lambda x: x[0], reverse=True)

    top_evals = pre_evals[:POP_SIZE]
    pop_pools = [e[1] for e in top_evals]
    pop_weights = [e[2] for e in top_evals]

    best_fit = top_evals[0][0]
    best_pool = top_evals[0][1]
    best_w = top_evals[0][2]

    # --- GA LOOP ---
    for _ in range(1, N_GENERATIONS + 1):
        s_vals, w_vals = optimize_population_gpu(pop_pools, pop_weights)
        evals = sorted([(float(s_vals[i]), pop_pools[i], w_vals[i]) for i in range(POP_SIZE)], key=lambda x: x[0], reverse=True)

        if evals[0][0] > best_fit:
            best_fit, best_pool, best_w = evals[0]

        elites = evals[:ELITISM]
        new_pools = [e[1] for e in elites]
        new_weights = [e[2] for e in elites]
        while len(new_pools) < POP_SIZE:
            if random.random() < CROSSOVER_RATE:
                p1 = random.choice(elites)[1]
                p2 = random.choice(elites)[1]
                new_pools.append(crossover_pool(p1, p2))
                new_weights.append(None)
            elif random.random() < CROSSOVER_RATE + MUTATION_RATE:
                p1 = random.choice(elites)[1]
                new_pools.append(mutate_pool(p1))
                new_weights.append(None)
            else:
                new_pools.append(random_pool())
                new_weights.append(None)
        pop_pools = new_pools
        pop_weights = new_weights

    return best_pool, best_w, best_fit

# -----------------------------
# 5. MAIN LOOP
# -----------------------------
if __name__ == "__main__":
    GLOBAL_BEST = 4.51
    print(f"🌋 STARTING OFFLINE MINING (Target: {GLOBAL_BEST})")

    if not os.path.exists(DATA_LAKE_DIR):
        print("❌ DATA_LAKE directory missing. Please initialize data.")
        sys.exit()

    try:
        for cycle in range(1, N_CYCLES + 1):
            print(f"\n⛏️  CYCLE {cycle}...")
            bp, bw, bs = run_cycle(cycle)

            # If cycle valid
            if bp is None or bw is None:
                print(f"⚠️  Invalid Cycle/Skip (Sharpe={bs:.4f}).")
                continue

            # Display every cycle result
            res = pd.DataFrame({"Ticker": bp, "Weight": bw}).sort_values("Weight", ascending=False)
            res = res[res["Weight"] > 0.001]
            print(f"📌 Best portfolio cycle {cycle} | Sharpe: {bs:.4f} | Record: {GLOBAL_BEST:.4f}")
            print(res)

            # Record Breaker -> Save + HOF
            if bs > GLOBAL_BEST:
                GLOBAL_BEST = bs
                print(f"\n🏆 NEW RECORD : {GLOBAL_BEST:.4f} 🏆")

                res.to_csv(CHAMPION_FILE, index=False)

                # HALL OF FAME
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Creating a string representation of the portfolio
                p_str = "|".join([f"{r.Ticker}:{r.Weight:.4f}" for r in res.itertuples()])
                log_entry = pd.DataFrame([{"Date": ts, "Cycle": cycle, "Sharpe": bs, "Portfolio": p_str}])
                
                # Check if file exists to determine if header is needed
                header_needed = not os.path.exists(HISTORY_FILE)
                log_entry.to_csv(HISTORY_FILE, mode='a', header=header_needed, index=False)

    except KeyboardInterrupt:
        print("\nManual Stop.")
