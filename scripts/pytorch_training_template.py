"""可直接运行的 PyTorch 二分类训练模板。需要先安装 torch。"""

from __future__ import annotations


def main() -> None:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:
        raise SystemExit("未安装 PyTorch。请先按官方安装器安装 torch。") from error

    torch.manual_seed(42)
    features = torch.randn(1_000, 2)
    targets = ((features[:, 0] ** 2 + features[:, 1]) > 0.5).float().unsqueeze(1)
    train_x, test_x = features[:800], features[800:]
    train_y, test_y = targets[:800], targets[800:]

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)
    model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(30):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            logits = model(batch_x)
            loss = loss_function(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
        if epoch % 5 == 0:
            print(f"epoch={epoch:02d} loss={total_loss / len(train_x):.4f}")

    model.eval()
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(test_x))
        predictions = (probabilities >= 0.5).float()
        accuracy = (predictions == test_y).float().mean()
    print(f"test_accuracy={accuracy.item():.3f}")
    torch.save({"model_state": model.state_dict()}, "binary_mlp.pt")


if __name__ == "__main__":
    main()
