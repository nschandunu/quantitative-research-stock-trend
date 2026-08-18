import os
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "AAPL_1y_daily.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "aapl_tcn.pt"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "aapl_scaler.joblib"
)

FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "ATR",
    "EMA20",
    "MOM6",
    "CCI",
    "MACD"
]

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("AAPL TCN PREDICTION")
print("=" * 60)

print("Device:", device)


class TemporalBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        dropout=0.2
    ):
        super().__init__()

        padding = (
            kernel_size - 1
        ) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            dropout
        )

        self.residual = (
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):

        residual = self.residual(x)

        out = self.conv1(x)
        out = out[:, :, :x.size(2)]
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, :x.size(2)]
        out = self.relu(out)
        out = self.dropout(out)

        return self.relu(
            out + residual
        )


class AAPLTCN(nn.Module):

    def __init__(
        self,
        num_features=9,
        num_classes=2
    ):
        super().__init__()

        self.tcn = nn.Sequential(

            TemporalBlock(
                in_channels=num_features,
                out_channels=32,
                kernel_size=3,
                dilation=1,
                dropout=0.2
            ),

            TemporalBlock(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                dilation=2,
                dropout=0.2
            )
        )

        self.classifier = nn.Linear(
            32,
            num_classes
        )

    def forward(self, x):

        x = x.transpose(1, 2)

        x = self.tcn(x)

        x = x[:, :, -1]

        return self.classifier(x)


aapl = pd.read_csv(
    DATA_PATH
)

aapl["Date"] = pd.to_datetime(
    aapl["Date"]
)

aapl = (
    aapl
    .sort_values("Date")
    .reset_index(drop=True)
)

print("\nData loaded:")

print(
    "Rows:",
    len(aapl)
)

print(
    "Date range:",
    aapl["Date"].min().date(),
    "→",
    aapl["Date"].max().date()
)

assert set(
    ["Date", "Open", "High", "Low", "Close", "Volume"]
).issubset(aapl.columns)

aapl["EMA20"] = (
    aapl["Close"]
    .ewm(
        span=20,
        adjust=False
    )
    .mean()
)

aapl["MOM6"] = (
    aapl["Close"]
    - aapl["Close"].shift(6)
)

previous_close = (
    aapl["Close"].shift(1)
)

true_range = pd.concat(
    [
        aapl["High"] - aapl["Low"],

        (
            aapl["High"]
            - previous_close
        ).abs(),

        (
            aapl["Low"]
            - previous_close
        ).abs()
    ],
    axis=1
).max(axis=1)

aapl["ATR"] = (
    true_range
    .rolling(window=14)
    .mean()
)

typical_price = (
    aapl["High"]
    + aapl["Low"]
    + aapl["Close"]
) / 3

cci_period = 20

tp_mean = (
    typical_price
    .rolling(cci_period)
    .mean()
)

mean_deviation = (
    typical_price
    .rolling(cci_period)
    .apply(
        lambda x: np.mean(
            np.abs(x - np.mean(x))
        ),
        raw=True
    )
)

aapl["CCI"] = (
    (typical_price - tp_mean)
    / (0.015 * mean_deviation)
)

ema12 = (
    aapl["Close"]
    .ewm(
        span=12,
        adjust=False
    )
    .mean()
)

ema26 = (
    aapl["Close"]
    .ewm(
        span=26,
        adjust=False
    )
    .mean()
)

aapl["MACD"] = (
    ema12 - ema26
)

print("\nIndicators calculated.")

print(
    aapl[
        [
            "ATR",
            "EMA20",
            "MOM6",
            "CCI",
            "MACD"
        ]
    ].isna().sum()
)

aapl_features = (
    aapl
    .dropna(subset=FEATURE_COLUMNS)
    .reset_index(drop=True)
)

assert len(aapl_features) >= 22

latest_window = (
    aapl_features[
        FEATURE_COLUMNS
    ]
    .tail(22)
    .values
)

latest_date = (
    aapl_features["Date"]
    .iloc[-1]
)

X_latest = np.asarray(
    latest_window,
    dtype=np.float32
)

X_latest = X_latest.reshape(
    1,
    22,
    9
)

print("\nLatest model input:")
print("Date:", latest_date.date())
print("Shape:", X_latest.shape)

assert X_latest.shape == (1, 22, 9)

scaler = joblib.load(
    SCALER_PATH
)

X_latest_2d = X_latest.reshape(
    -1,
    9
)

X_latest_scaled_2d = (
    scaler.transform(
        X_latest_2d
    )
)

X_latest_scaled = (
    X_latest_scaled_2d
    .reshape(1, 22, 9)
    .astype(np.float32)
)

assert np.isfinite(
    X_latest_scaled
).all()

print("\nScaler loaded.")
print(
    "Scaler type:",
    type(scaler).__name__
)
print("Scaling: PASSED")


model = AAPLTCN(
    num_features=9,
    num_classes=2
).to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("\nTCN model loaded.")
print("Model path:", MODEL_PATH)
print("Model loading: PASSED")

X_tensor = torch.tensor(
    X_latest_scaled,
    dtype=torch.float32,
    device=device
)

with torch.no_grad():

    logits = model(
        X_tensor
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    prediction = torch.argmax(
        probabilities,
        dim=1
    ).item()

fall_probability = (
    probabilities[0, 0]
    .item()
)

rise_probability = (
    probabilities[0, 1]
    .item()
)

signal = {
    0: "Fall",
    1: "Rise"
}[prediction]

confidence = max(
    fall_probability,
    rise_probability
) * 100

print("\n" + "=" * 60)
print("AAPL TCN PREDICTION RESULT")
print("=" * 60)

print(
    "Prediction date:",
    latest_date.date()
)

print(
    "Signal:",
    signal
)

print(
    f"Fall probability: {fall_probability:.4f}"
)

print(
    f"Rise probability: {rise_probability:.4f}"
)

print(
    f"Confidence: {confidence:.2f}%"
)

print("=" * 60)
print("Prediction complete.")
print("=" * 60)

assert latest_date == aapl_features["Date"].iloc[-1]

assert X_latest.shape == (1, 22, 9)

assert np.isfinite(X_latest_scaled).all()

assert prediction in (0, 1)

assert abs(
    fall_probability + rise_probability - 1.0
) < 1e-6

assert signal in ("Fall", "Rise")

print("\nmain.py validation: PASSED")