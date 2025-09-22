# linear_noslide_with_test.py
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
import joblib
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# ----------------- 用户可修改参数 -----------------
npy_path = "D:\jz_pack_data\pack_json\combined_cells.npy"   # (n_series, T_max)
in_len = 12
batch_size = 64
epochs = 80
lr = 1e-3
val_ratio = 0.1
test_ratio = 0.1
device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42
scaler_path = "./save_model/scaler.pkl"
model_path = "./save_model/best_model.pth"
# --------------------------------------------------

# 固定随机种子
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

class SingleSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, in_len: int):
        self.data = data.astype(np.float32)
        self.n_series, self.T = self.data.shape
        self.in_len = in_len
        if in_len >= self.T:
            raise ValueError("in_len must be smaller than sequence length T")
        self.out_len = self.T - in_len

    def __len__(self):
        return self.n_series

    def __getitem__(self, idx):
        serie = self.data[idx]
        x = serie[:self.in_len]
        y = serie[self.in_len:]
        return x, y

class LinearModel(nn.Module):
    def __init__(self, in_len:int, out_len:int):
        super().__init__()
        self.fc = nn.Linear(in_len, out_len)
    def forward(self, x):
        return self.fc(x)

def train_and_eval():
    # load
    arr = np.load(npy_path)   # (n_series, T)
    n_series, T = arr.shape
    if in_len >= T:
        raise RuntimeError("in_len must be < T")
    out_len = T - in_len
    print("data shape:", arr.shape, "in_len:", in_len, "out_len:", out_len)

    # split indices for train/val/test
    indices = np.arange(n_series)
    np.random.shuffle(indices)
    n_test = max(1, int(n_series * test_ratio))
    n_val = max(1, int(n_series * val_ratio))
    # ensure non-overlap, test first, then val, rest train
    test_idx = indices[:n_test].tolist()
    val_idx = indices[n_test:n_test + n_val].tolist()
    train_idx = indices[n_test + n_val:].tolist()
    print(f"n_train={len(train_idx)}, n_val={len(val_idx)}, n_test={len(test_idx)}")

    # fit scaler on train only (flatten all train values)
    scaler = StandardScaler()
    train_data = arr[train_idx]   # (n_train, T)
    scaler.fit(train_data.reshape(-1, 1))
    joblib.dump(scaler, scaler_path)
    print("scaler fitted on train and saved to", scaler_path)

    # transform whole dataset using same scaler
    arr_scaled = scaler.transform(arr.reshape(-1,1)).reshape(arr.shape)

    # datasets / loaders
    ds = SingleSeriesDataset(arr_scaled, in_len=in_len)
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    test_ds = Subset(ds, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # model
    model = LinearModel(in_len=in_len, out_len=out_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = float('inf')
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        nb = 0
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            pred = model(xb)
            loss = criterion(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            nb += 1
        train_loss = train_loss / max(1, nb)

        # val
        model.eval()
        val_loss = 0.0
        vb = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device).float(); yb = yb.to(device).float()
                pred = model(xb)
                val_loss += criterion(pred, yb).item()
                vb += 1
        val_loss = val_loss / max(1, vb)
        print(f"Epoch {epoch}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "scaler_path": scaler_path,
                "in_len": in_len,
                "out_len": out_len
            }, model_path)
            print(" saved best model to", model_path)

    print("training finished. best_val:", best_val)

    # load best model
    ck = torch.load(model_path, map_location=device)
    model.load_state_dict(ck["model_state"])
    model.to(device)
    model.eval()

    # ----- 只在 test 集上预测并反归一化 -----
    y_true_all = []
    y_pred_all = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device).float()
            pred_scaled = model(xb).cpu().numpy()    # (batch, out_len)
            y_scaled = yb.numpy()                    # (batch, out_len)
            # inverse transform: scaler works on 2D flattened arrays
            pred_orig = scaler.inverse_transform(pred_scaled.reshape(-1,1)).reshape(pred_scaled.shape)
            y_orig = scaler.inverse_transform(y_scaled.reshape(-1,1)).reshape(y_scaled.shape)
            y_true_all.append(y_orig)
            y_pred_all.append(pred_orig)

    if len(y_true_all) == 0:
        raise RuntimeError("test set empty or no predictions.")

    y_true_all = np.vstack(y_true_all)   # (n_test, out_len)
    y_pred_all = np.vstack(y_pred_all)

    # metrics (按样本平均)
    mse = mean_squared_error(y_true_all.ravel(), y_pred_all.ravel())
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true_all.ravel(), y_pred_all.ravel())
    print(f"TEST MSE={mse:.6f} RMSE={rmse:.6f} MAE={mae:.6f}")

    # 保存 test 预测与真实值
    np.save("y_test.npy", y_true_all)
    np.save("preds_test.npy", y_pred_all)
    print("saved y_test.npy and preds_test.npy")

    return model, scaler, (y_true_all, y_pred_all)

if __name__ == "__main__":
    train_and_eval()
