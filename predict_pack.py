import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from load_pack_data import *
from sklearn.preprocessing import StandardScaler
from models.deepset import DeepSetModel
from models.packchainGNN import PackChainGraphModel
from models.packTransformer import PackTransformer

class PackCellsDataset(Dataset):
    def __init__(self, X_tensor, y_arr):
        self.X = torch.from_numpy(X_tensor).float()   # (N, 114, m)
        self.y = torch.from_numpy(y_arr).float()      # (N, )

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]





def train_single_target(X_tensor, y_arr, target_idx=0,
                                epochs=100, batch_size=32, lr=1e-3,
                                val_split=0.1, device=None, model='Graph'):
    """
    训练模型，只针对一个目标特征 (target_idx)
    自动对输入特征和输出目标做标准化，训练结束后返回反归一化预测。
    """

    # 自动检测 GPU
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 只选择一个目标
    y_target = y_arr[:, target_idx].reshape(-1, 1)

    # ----------------------
    #输入特征归一化
    N, n_cells, m = X_tensor.shape
    X_reshaped = X_tensor.reshape(N * n_cells, m)
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X_reshaped)
    X_tensor_scaled = X_scaled.reshape(N, n_cells, m)

    #输出目标归一化
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y_target)

    # ----------------------
    # 划分训练/验证集
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tensor_scaled, y_scaled, test_size=val_split, random_state=42
    )
    ds_tr = PackCellsDataset(X_tr, y_tr)
    ds_val = PackCellsDataset(X_val, y_val)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)

    if model == 'Graph':
        model = PackChainGraphModel(in_dim=m,
                                    hidden_dim=64,
                                    out_dim=1,
                                    n_cells=n_cells,
                                    gnn_type="gat",
                                    readout_type='maxmean').to(device)
    elif model == 'DeepSet':
        model = DeepSetModel(in_dim=m,
                             emb_dim=128,
                             hidden_dim=256,
                             out_dim=1,
                             agg='mean').to(device)
    elif model == 'Transformer':
        model = PackTransformer(in_dim=m,
                                model_dim=192,
                                num_layers=2,
                                num_heads=4,
                                out_dim=1,
                                n_cells=n_cells,
                                max_rel=4,
                                use_pack_token=True).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience = 50
    cur_pat = 0

    for ep in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device).squeeze(-1)  # (B,)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses))

        # 验证
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(device)
                yb = yb.to(device).squeeze(-1)
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())
        val_loss = float(np.mean(val_losses))

        print(f"Epoch {ep} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            cur_pat = 0
            torch.save(model.state_dict(), f"deepset_best_target{target_idx}.pth")
        else:
            cur_pat += 1
            if cur_pat >= patience:
                print("Early stopping.")
                break

    # load best
    model.load_state_dict(torch.load(f"deepset_best_target{target_idx}.pth", map_location=device))

    # ----------------------
    # 计算指标（反归一化）
    model.eval()
    preds_scaled = []
    trues_scaled = []
    with torch.no_grad():
        for xb, yb in dl_val:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds_scaled.append(pred)
            trues_scaled.append(yb.numpy())
    preds_scaled = np.concatenate(preds_scaled, axis=0).flatten()
    trues_scaled = np.concatenate(trues_scaled, axis=0).flatten()

    # 反归一化
    preds = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    trues = y_scaler.inverse_transform(trues_scaled.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)
    mape = np.mean(np.abs((trues - preds) / (trues + 1e-8))) * 100

    print(f"[DeepSet] target#{target_idx} rmse={rmse:.4f} mae={mae:.4f} mape={mape:.4f} r2={r2:.4f}")

    return model, X_scaler, y_scaler


def main():
    PACK_MAP_JSON = "./dataset/xz2_pack_data/4000005_pack_data/0BT/0BT_pack_cells_features.json"
    cell_feature_names = [
        "Cell_Capacity", "Cell_OCV4", "Cell_OCV5", #"Cell_K",
        "Cell_InR4", "Cell_InR5", "Cell_Times3"
    ]

    # 要预测的 pack-level 目标（示例，和你的 pack_feature 字段对应）
    pack_target_names = [
        "13|压差计算(mV)-充电末端压差",
        "1|压差计算(mV)-静态末端压差",
        "5|压差计算(mV)-充电末端压差",
        "6|压差计算(mV)-静态末端压差",
        "7|压差计算(mV)-放电末端压差",
        "7|容量计算(AH)-阶段放电容量",
        "8|压差计算(mV)-静态末端压差"
    ]

    N_CELLS_EXPECTED = 114
    print("加载 pack_data ...")
    with open(PACK_MAP_JSON, "r", encoding="utf-8") as f:
        pack_map = json.load(f)
        del pack_map["03HPB0BT0001EYF260000102"]

    print("构建数据集...")
    X_tensor, X_agg_df, y_df, index_to_pack = build_dataset_from_pack_map(
        pack_map, cell_feature_names, pack_target_names, n_cells=N_CELLS_EXPECTED
    )

    print(f"找到 {len(X_agg_df)} 个可用的 pack，用于训练。")

    # X_tensor.shape = (N, 114, m)，y_arr.shape = (N, k)
    model = train_single_target(X_tensor, y_df.values, target_idx=5, epochs=5000, batch_size=32, model='Graph')


if __name__ == "__main__":
    main()