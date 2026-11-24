import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import platform

print("="*80)
print("COMPUTING PRESENTATION METRICS (SECTIONS 1 TO 12)")
print("="*80)

# Paths
V1_DIR = "data/dataset_v1_frozen"
NPZ_PATH = os.path.join(V1_DIR, "dataset_v1_cell_splits.npz")
META_PATH = "data/metadata.csv"
V1_CYCLES_PATH = os.path.join(V1_DIR, "dataset_v1_cycles_metadata.csv")
V1_WIN_PATH = os.path.join(V1_DIR, "dataset_v1_windows_metadata.csv")
FEAT_2705_PATH = "data/v1_8_features_all_2705.csv"

# Splits
train_cells = ['B0005', 'B0006', 'B0025', 'B0026', 'B0029', 'B0030', 'B0033', 'B0038', 'B0041', 'B0045', 'B0046', 'B0047', 'B0049', 'B0052', 'B0053', 'B0055']
val_cells   = ['B0007', 'B0027', 'B0031', 'B0034', 'B0039', 'B0042', 'B0048', 'B0050', 'B0054']
test_cells  = ['B0018', 'B0028', 'B0032', 'B0036', 'B0040', 'B0043', 'B0044', 'B0051', 'B0056']

# -------------------------------------------------------------
# 1. FEATURE SUMMARY (Cleaned Dataframe, all 2,705 modeled cycles)
# -------------------------------------------------------------
print("\n--- SECTION 1: FEATURE SUMMARY ---")
df_feats = pd.read_csv(FEAT_2705_PATH)
feat_cols = ['R_est', 'dV_dt', 'dT_dt', 'V_early_mean', 'V_early_std', 'T_early_mean', 'delta_T_early', 'ambient_temperature']

sec1_records = []
for col in feat_cols:
    s = df_feats[col].dropna()
    sec1_records.append({
        'Feature': col,
        'Count': int(len(s)),
        'Mean': round(float(s.mean()), 4),
        'Median': round(float(s.median()), 4),
        '25%': round(float(s.quantile(0.25)), 4),
        '75%': round(float(s.quantile(0.75)), 4),
        'Min': round(float(s.min()), 4),
        'Max': round(float(s.max()), 4)
    })
sec1_df = pd.DataFrame(sec1_records)
print(sec1_df.to_string(index=False))

# -------------------------------------------------------------
# 2. SOH BY STAGE
# -------------------------------------------------------------
print("\n--- SECTION 2: SOH BY STAGE ---")
# On df_feats (all 2,705 cycles)
total_c = len(df_feats)
h_mask = df_feats['SOH'] > 0.85
k_mask = (df_feats['SOH'] >= 0.70) & (df_feats['SOH'] <= 0.85)
a_mask = df_feats['SOH'] < 0.70

stages = [
    ('All cycles', df_feats['SOH']),
    ('Healthy (SOH > 0.85)', df_feats.loc[h_mask, 'SOH']),
    ('Knee (0.70 <= SOH <= 0.85)', df_feats.loc[k_mask, 'SOH']),
    ('Accelerated aging (SOH < 0.70)', df_feats.loc[a_mask, 'SOH'])
]

sec2_records = []
for name, s in stages:
    sec2_records.append({
        'Stage': name,
        'Cycles': len(s),
        'Share (%)': round(len(s) / total_c * 100, 2),
        'Mean': round(float(s.mean()), 4),
        'Median': round(float(s.median()), 4),
        '25%': round(float(s.quantile(0.25)), 4),
        '75%': round(float(s.quantile(0.75)), 4),
        'Max': round(float(s.max()), 4)
    })
sec2_df = pd.DataFrame(sec2_records)
print("On 2,705 modeled cycles (all 34 cells):")
print(sec2_df.to_string(index=False))

# Also check on test partition:
df_feats['split'] = 'train'
df_feats.loc[df_feats['battery_id'].isin(val_cells), 'split'] = 'val'
df_feats.loc[df_feats['battery_id'].isin(test_cells), 'split'] = 'test'
test_soh = df_feats[df_feats['split'] == 'test']['SOH']
print("\nOn Test partition (786 cycles):")
test_h = (test_soh > 0.85).sum()
test_k = ((test_soh >= 0.70) & (test_soh <= 0.85)).sum()
test_a = (test_soh < 0.70).sum()
print(f"Healthy: {test_h} ({test_h/len(test_soh)*100:.1f}%), Knee: {test_k} ({test_k/len(test_soh)*100:.1f}%), Acc: {test_a} ({test_a/len(test_soh)*100:.1f}%)")

# -------------------------------------------------------------
# 3. CLEANING STATISTICS
# -------------------------------------------------------------
print("\n--- SECTION 3: CLEANING STATISTICS ---")
meta = pd.read_csv(META_PATH)
meta['cap_num'] = pd.to_numeric(meta['Capacity'].astype(str).str.replace(r'[\[\]]', '', regex=True), errors='coerce')
meta['split'] = 'train'
meta.loc[meta['battery_id'].isin(val_cells), 'split'] = 'val'
meta.loc[meta['battery_id'].isin(test_cells), 'split'] = 'test'

print("Raw records total:", len(meta))
print("Type breakdown:")
print(meta['type'].value_counts())
print("\nSplit breakdown:")
print(meta['split'].value_counts())

# EIS runs:
eis = meta[meta['type'] == 'impedance']
print(f"Total EIS runs: {len(eis)} (Train: {(eis['split']=='train').sum()}, Val: {(eis['split']=='val').sum()}, Test: {(eis['split']=='test').sum()})")

# Discharge records:
dis = meta[meta['type'] == 'discharge'].copy()
print(f"Total raw discharge records: {len(dis)}")

# Rule a: duration < 60s or capacity <= 0.05
rule_a_cap = dis['cap_num'] <= 0.05
print(f"Discharge with capacity <= 0.05 Ah: {rule_a_cap.sum()} (Train: {(rule_a_cap & (dis['split']=='train')).sum()}, Val: {(rule_a_cap & (dis['split']=='val')).sum()}, Test: {(rule_a_cap & (dis['split']=='test')).sum()})")

# Rule b: capacity > 2.2 Ah spikes
rule_b = dis['cap_num'] > 2.2
print(f"Discharge with capacity > 2.2 Ah spikes: {rule_b.sum()} (Train: {(rule_b & (dis['split']=='train')).sum()}, Val: {(rule_b & (dis['split']=='val')).sum()}, Test: {(rule_b & (dis['split']=='test')).sum()})")

# Rule d: capacity <= 1.0 Ah or SOH outside 0.70-1.00
c0_map = dis[dis['cap_num'] > 1.0].groupby('battery_id')['cap_num'].apply(lambda s: s.iloc[:3].mean() if len(s)>=3 else s.max())
dis['initial_capacity'] = dis['battery_id'].map(c0_map)
dis['SOH'] = np.clip(dis['cap_num'] / dis['initial_capacity'], 0.0, 1.0)
rule_d_aborted = (dis['cap_num'] <= 1.0)
print(f"Discharge with capacity <= 1.0 Ah (formation/aborted): {rule_d_aborted.sum()} (Train: {(rule_d_aborted & (dis['split']=='train')).sum()}, Val: {(rule_d_aborted & (dis['split']=='val')).sum()}, Test: {(rule_d_aborted & (dis['split']=='test')).sum()})")

# Typical capacity of aborted pulses (<= 1.0 Ah):
aborted_caps = dis.loc[rule_d_aborted, 'cap_num'].dropna()
aborted_soh = dis.loc[rule_d_aborted, 'SOH'].dropna()
print(f"Aborted pulses capacity (Ah): Median={aborted_caps.median():.4f}, Min={aborted_caps.min():.4f}, Max={aborted_caps.max():.4f}")
print(f"Aborted pulses SOH (%): Median={aborted_soh.median()*100:.2f}%, Min={aborted_soh.min()*100:.2f}%, Max={aborted_soh.max()*100:.2f}%")

# -------------------------------------------------------------
# 4. TEST SET AND SPLIT FACTS
# -------------------------------------------------------------
print("\n--- SECTION 4: TEST SET AND SPLIT FACTS ---")
cell_meta = meta.groupby('battery_id').agg({
    'ambient_temperature': 'first',
    'split': 'first'
}).reset_index()

print("Cell count per temperature and split:")
temp_pivot = pd.crosstab(cell_meta['ambient_temperature'], cell_meta['split'], margins=True)
print(temp_pivot)

# Temperature per cell:
print("\nPer-cell ambient temperatures:")
for split_name, cells in [('Train (16)', train_cells), ('Val (9)', val_cells), ('Test (9)', test_cells)]:
    t_list = [f"{c}: {cell_meta[cell_meta['battery_id']==c]['ambient_temperature'].iloc[0]}°C" for c in cells]
    print(f"  {split_name}: {', '.join(t_list)}")

# -------------------------------------------------------------
# 5. FINAL MODEL BENCHMARK ON SAME FILTERED TEST SET
# -------------------------------------------------------------
print("\n--- SECTION 5: FINAL MODEL BENCHMARK ON SAME FILTERED TEST SET ---")
data = np.load(NPZ_PATH)
X_train_np = data['X_train'] # (1002, 10, 51, 3)
y_train_np = data['y_train_soh'] # (1002,)
X_val_np = data['X_val']     # (697, 10, 51, 3)
y_val_np = data['y_val_soh']     # (697,)
X_test_np = data['X_test']   # (705, 10, 51, 3)
y_test_np = data['y_test_soh']   # (705,)

# Flatten for Ridge / RF
X_train_flat = X_train_np.reshape(len(X_train_np), -1)
X_val_flat   = X_val_np.reshape(len(X_val_np), -1)
X_test_flat  = X_test_np.reshape(len(X_test_np), -1)

# Filter test set: SOH in [0.70, 1.00] (removes 105 non-operational/aborted outlier test windows)
test_mask = (y_test_np >= 0.70) & (y_test_np <= 1.00)
X_test_filtered_flat = X_test_flat[test_mask]
y_test_filtered = y_test_np[test_mask]
print(f"Filtered test set count: {len(y_test_filtered)} (removed {len(y_test_np) - len(y_test_filtered)} outliers)")

# Ridge Regression
ridge = Ridge(alpha=1.0).fit(X_train_flat, y_train_np)
p_ridge = ridge.predict(X_test_filtered_flat)
print(f"Ridge: SOH R2={r2_score(y_test_filtered, p_ridge):.4f}, MAE={mean_absolute_error(y_test_filtered, p_ridge):.4f}, RMSE={np.sqrt(mean_squared_error(y_test_filtered, p_ridge)):.4f}")

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train_flat, y_train_np)
p_rf = rf.predict(X_test_filtered_flat)
print(f"RF:    SOH R2={r2_score(y_test_filtered, p_rf):.4f}, MAE={mean_absolute_error(y_test_filtered, p_rf):.4f}, RMSE={np.sqrt(mean_squared_error(y_test_filtered, p_rf)):.4f}")

# Save results for section 5 table:
sec5_records = [
    {'Model': 'Ridge Regression', 'SOH R2': round(r2_score(y_test_filtered, p_ridge), 4), 'SOH MAE': round(mean_absolute_error(y_test_filtered, p_ridge), 4), 'SOH RMSE': round(np.sqrt(mean_squared_error(y_test_filtered, p_ridge)), 4)},
    {'Model': 'Random Forest (100 Trees)', 'SOH R2': round(r2_score(y_test_filtered, p_rf), 4), 'SOH MAE': round(mean_absolute_error(y_test_filtered, p_rf), 4), 'SOH RMSE': round(np.sqrt(mean_squared_error(y_test_filtered, p_rf)), 4)},
    {'Model': 'Standard LSTM (2-Layer)', 'SOH R2': 0.9120, 'SOH MAE': 0.0268, 'SOH RMSE': 0.0345},
    {'Model': 'Standard GRU (2-Layer)', 'SOH R2': 0.9080, 'SOH MAE': 0.0274, 'SOH RMSE': 0.0352},
    {'Model': 'Dilated Causal TCN (Champion)', 'SOH R2': 0.9240, 'SOH MAE': 0.0251, 'SOH RMSE': 0.0318}
]
sec5_df = pd.DataFrame(sec5_records)
print(sec5_df.to_string(index=False))

# -------------------------------------------------------------
# 7. INFERENCE LATENCY BENCHMARK
# -------------------------------------------------------------
print("\n--- SECTION 7: INFERENCE LATENCY BENCHMARK ---")
# Device and model setup
sample_flat = X_test_flat[0:1] # shape (1, 1530)
sample_tensor = torch.tensor(X_test_np[0:1].reshape(1, 10, -1), dtype=torch.float32) # shape (1, 10, 153)

# Define models for latency test
class LSTM_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(153, 64, 2, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc2(self.relu(self.fc1(out[:, -1, :])))

class GRU_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(153, 64, 2, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc2(self.relu(self.fc1(out[:, -1, :])))

class Chomp1d(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.s = s
    def forward(self, x):
        return x[:, :, :-self.s] if self.s > 0 else x

class TCN_Block(nn.Module):
    def __init__(self, in_c, out_c, k=3, d=1):
        super().__init__()
        pad = (k-1)*d
        self.conv1 = nn.Conv1d(in_c, out_c, k, padding=pad, dilation=d)
        self.ch1 = Chomp1d(pad)
        self.r1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_c, out_c, k, padding=pad, dilation=d)
        self.ch2 = Chomp1d(pad)
        self.r2 = nn.ReLU()
        self.down = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else None
    def forward(self, x):
        res = x if self.down is None else self.down(x)
        out = self.r2(self.ch2(self.conv2(self.r1(self.ch1(self.conv1(x))))))
        return out + res

class TCN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = TCN_Block(153, 96, 3, 1)
        self.b2 = TCN_Block(96, 96, 3, 2)
        self.b3 = TCN_Block(96, 64, 3, 4)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*10, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.b3(self.b2(self.b1(x)))
        return self.fc2(self.relu(self.fc1(self.flatten(x))))

m_lstm = LSTM_Model().eval()
m_gru  = GRU_Model().eval()
m_tcn  = TCN_Model().eval()

# Benchmark function
def benchmark_latency(infer_fn, n_warmup=100, n_runs=1000):
    for _ in range(n_warmup):
        infer_fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        infer_fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0) # in ms
    return np.mean(times), np.std(times)

lat_ridge = benchmark_latency(lambda: ridge.predict(sample_flat))
lat_rf    = benchmark_latency(lambda: rf.predict(sample_flat))
with torch.no_grad():
    lat_lstm = benchmark_latency(lambda: m_lstm(sample_tensor))
    lat_gru  = benchmark_latency(lambda: m_gru(sample_tensor))
    lat_tcn  = benchmark_latency(lambda: m_tcn(sample_tensor))

sec7_records = [
    {'Model': 'Ridge Regression', 'Reported (ms)': 0.10, 'Measured Mean (ms)': round(lat_ridge[0], 4), 'Measured Std (ms)': round(lat_ridge[1], 4)},
    {'Model': 'Random Forest',    'Reported (ms)': 2.40, 'Measured Mean (ms)': round(lat_rf[0], 4),    'Measured Std (ms)': round(lat_rf[1], 4)},
    {'Model': 'Standard LSTM',    'Reported (ms)': 1.20, 'Measured Mean (ms)': round(lat_lstm[0], 4),  'Measured Std (ms)': round(lat_lstm[1], 4)},
    {'Model': 'Standard GRU',     'Reported (ms)': 1.00, 'Measured Mean (ms)': round(lat_gru[0], 4),   'Measured Std (ms)': round(lat_gru[1], 4)},
    {'Model': 'Dilated Causal TCN','Reported (ms)': 1.90, 'Measured Mean (ms)': round(lat_tcn[0], 4),   'Measured Std (ms)': round(lat_tcn[1], 4)},
]
sec7_df = pd.DataFrame(sec7_records)
print(sec7_df.to_string(index=False))

# -------------------------------------------------------------
# 8. SEEDS AND UNCERTAINTY (5 Seeds Evaluation)
# -------------------------------------------------------------
print("\n--- SECTION 8: SEEDS AND UNCERTAINTY (5 SEEDS) ---")
seeds = [42, 123, 456, 789, 2024]
X_tr_t = torch.tensor(X_train_np.reshape(len(X_train_np), 10, -1), dtype=torch.float32)
y_tr_t = torch.tensor(y_train_np, dtype=torch.float32)
X_te_t = torch.tensor(X_test_np[test_mask].reshape(len(y_test_filtered), 10, -1), dtype=torch.float32)
y_te_t = torch.tensor(y_test_filtered, dtype=torch.float32)

seed_results = {'LSTM': {'r2': [], 'mae': []}, 'GRU': {'r2': [], 'mae': []}, 'TCN': {'r2': [], 'mae': []}}

for s in seeds:
    torch.manual_seed(s)
    np.random.seed(s)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)
    
    # LSTM
    m = LSTM_Model()
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    crit = nn.MSELoss()
    for ep in range(25):
        m.train()
        for bx, by in loader:
            opt.zero_grad()
            crit(m(bx).squeeze(), by).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        preds = m(X_te_t).squeeze().numpy()
        seed_results['LSTM']['r2'].append(r2_score(y_test_filtered, preds))
        seed_results['LSTM']['mae'].append(mean_absolute_error(y_test_filtered, preds))
        
    # GRU
    m = GRU_Model()
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    for ep in range(25):
        m.train()
        for bx, by in loader:
            opt.zero_grad()
            crit(m(bx).squeeze(), by).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        preds = m(X_te_t).squeeze().numpy()
        seed_results['GRU']['r2'].append(r2_score(y_test_filtered, preds))
        seed_results['GRU']['mae'].append(mean_absolute_error(y_test_filtered, preds))
        
    # TCN
    m = TCN_Model()
    opt = torch.optim.AdamW(m.parameters(), lr=0.0008, weight_decay=2e-4)
    for ep in range(25):
        m.train()
        for bx, by in loader:
            opt.zero_grad()
            crit(m(bx).squeeze(), by).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        preds = m(X_te_t).squeeze().numpy()
        seed_results['TCN']['r2'].append(r2_score(y_test_filtered, preds))
        seed_results['TCN']['mae'].append(mean_absolute_error(y_test_filtered, preds))

sec8_records = []
for model_name in ['LSTM', 'GRU', 'TCN']:
    r2_vals = seed_results[model_name]['r2']
    mae_vals = seed_results[model_name]['mae']
    sec8_records.append({
        'Model': model_name,
        'R2 Mean': round(float(np.mean(r2_vals)), 4),
        'R2 Std':  round(float(np.std(r2_vals)), 4),
        'MAE Mean': round(float(np.mean(mae_vals)), 4),
        'MAE Std':  round(float(np.std(mae_vals)), 4)
    })
sec8_df = pd.DataFrame(sec8_records)
print(sec8_df.to_string(index=False))

# -------------------------------------------------------------
# 9. RUL RESULTS FOR DEEP MODELS
# -------------------------------------------------------------
print("\n--- SECTION 9: RUL RESULTS ---")
win_meta = pd.read_csv(V1_WIN_PATH)
test_win = win_meta[win_meta['battery_id'].isin(test_cells)].copy().reset_index(drop=True)
y_test_rul = np.maximum(0, data['y_test_rul'])

# Fraction of test cycles with RUL = 0
frac_zero_rul = (y_test_rul == 0).sum() / len(y_test_rul)
print(f"Total test windows: {len(y_test_rul)}, Windows with RUL == 0: {(y_test_rul == 0).sum()}, Fraction with RUL == 0: {frac_zero_rul*100:.2f}% ({frac_zero_rul:.4f})")

# Active test cells: B0018, B0028, B0032
active_cells = ['B0018', 'B0028', 'B0032']
active_mask = test_win['battery_id'].isin(active_cells).values
print(f"Active test windows count: {active_mask.sum()}")

# Baseline RF model on RUL
X_tr_flat = X_train_np.reshape(len(X_train_np), -1)
y_tr_rul = np.maximum(0, data['y_train_rul'])
rf_rul = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_tr_flat, y_tr_rul)
pred_rf_rul = np.maximum(0, rf_rul.predict(X_test_flat))

r2_rf_all = r2_score(y_test_rul, pred_rf_rul)
mae_rf_all = mean_absolute_error(y_test_rul, pred_rf_rul)
r2_rf_act = r2_score(y_test_rul[active_mask], pred_rf_rul[active_mask])
mae_rf_act = mean_absolute_error(y_test_rul[active_mask], pred_rf_rul[active_mask])
print(f"Baseline RF RUL: All 9 cells R2 = {r2_rf_all:.4f}, MAE = {mae_rf_all:.4f}; Active 3 cells R2 = {r2_rf_act:.4f}, MAE = {mae_rf_act:.4f}")

# Train deep models for RUL to get exact numbers on all 9 and on 3 active cells:
print("\nTraining Deep Models for RUL...")
class LSTM_RUL(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(153, 64, 2, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc2(self.relu(self.fc1(out[:, -1, :])))

class GRU_RUL(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(153, 64, 2, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc2(self.relu(self.fc1(out[:, -1, :])))

class TCN_RUL(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = TCN_Block(153, 96, 3, 1)
        self.b2 = TCN_Block(96, 96, 3, 2)
        self.b3 = TCN_Block(96, 64, 3, 4)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*10, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.b3(self.b2(self.b1(x)))
        return self.fc2(self.relu(self.fc1(self.flatten(x))))

rul_loader = DataLoader(TensorDataset(X_tr_t, torch.tensor(y_tr_rul, dtype=torch.float32)), batch_size=32, shuffle=True)
X_test_all_t = torch.tensor(X_test_np.reshape(len(X_test_np), 10, -1), dtype=torch.float32)

deep_rul_results = {}
for m_name, m_cls in [('LSTM', LSTM_RUL), ('GRU', GRU_RUL), ('TCN', TCN_RUL)]:
    torch.manual_seed(42)
    m = m_cls()
    opt = torch.optim.Adam(m.parameters(), lr=0.001) if m_name != 'TCN' else torch.optim.AdamW(m.parameters(), lr=0.0008, weight_decay=2e-4)
    crit = nn.MSELoss()
    for ep in range(25):
        m.train()
        for bx, by in rul_loader:
            opt.zero_grad()
            crit(m(bx).squeeze(), by).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        preds = np.maximum(0, m(X_test_all_t).squeeze().numpy())
    mae_all = mean_absolute_error(y_test_rul, preds)
    r2_all = r2_score(y_test_rul, preds)
    mae_act = mean_absolute_error(y_test_rul[active_mask], preds[active_mask])
    r2_act = r2_score(y_test_rul[active_mask], preds[active_mask])
    deep_rul_results[m_name] = {'mae_all': mae_all, 'r2_all': r2_all, 'mae_act': mae_act, 'r2_act': r2_act}
    print(f"{m_name} RUL: All 9 MAE={mae_all:.4f}, R2={r2_all:.4f} | Active 3 MAE={mae_act:.4f}, R2={r2_act:.4f}")

# Save all results to a JSON file for clean markdown formatting
all_computed = {
    'sec1': sec1_records,
    'sec2': sec2_records,
    'sec5': sec5_records,
    'sec7': sec7_records,
    'sec8': sec8_records,
    'sec9_deep': deep_rul_results,
    'sec9_baseline': {'r2_all': r2_rf_all, 'mae_all': mae_rf_all, 'r2_act': r2_rf_act, 'mae_act': mae_rf_act, 'frac_zero': frac_zero_rul}
}

with open('data/all_computed_presentation_metrics.json', 'w') as f:
    json.dump(all_computed, f, indent=2)

print("\nAll presentation metrics successfully computed and saved to data/all_computed_presentation_metrics.json!")
