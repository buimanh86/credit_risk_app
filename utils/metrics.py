import pandas as pd
from scipy.stats import ks_2samp

def ks_statistic(y_true, y_prob):
    """
    Tính toán chỉ số Kolmogorov-Smirnov (KS) cho mô hình phân loại nhị phân.
    """
    df = pd.DataFrame({
        "y": y_true,
        "prob": y_prob
    })

    # Tách xác suất dự báo cho nhóm nợ xấu (1) và nhóm tốt (0)
    bad = df[df["y"] == 1]["prob"]
    good = df[df["y"] == 0]["prob"]

    # Sử dụng kiểm định ks_2samp để lấy giá trị statistic
    if len(bad) > 0 and len(good) > 0:
        ks = ks_2samp(bad, good).statistic
    else:
        ks = 0.0

    return ks