"""Explicit released-checkpoint evaluation profiles; no inferred timestamps."""

PROFILES = {
    "pems08": {
        "title": "PEMS08", "slug": "pems08", "shape": [170, 17856, 2],
        "stage1": "data/stage1/08.npy", "stage2": "data/stage2/08fine-tuning.npy",
        "checkpoint": "checkpoint/08/best_model_dict.pth", "interval_minutes": 5,
        "anchors": {
            "S1": {"MAE": 11.87, "RMSE": 18.53, "WMAPE_percent": 5.16},
            "S2": {"MAE": 14.78, "RMSE": 22.96, "WMAPE_percent": 6.43},
            "S3": {"MAE": 13.25, "RMSE": 20.39, "WMAPE_percent": 5.76}},
    },
    "bj": {
        "title": "Beijing Metro", "slug": "beijing", "shape": [276, 1800, 2],
        "stage1": "data/stage1/beijing.npy", "stage2": "data/stage2/bjfine-tuning.npy",
        "checkpoint": "checkpoint/bj/best_model_dict.pth", "interval_minutes": 15,
        "anchors": {
            "S1": {"MAE": 23.48, "RMSE": 42.06, "WMAPE_percent": 8.82},
            "S2": {"MAE": 34.74, "RMSE": 61.98, "WMAPE_percent": 12.95},
            "S3": {"MAE": 27.88, "RMSE": 48.17, "WMAPE_percent": 10.52}},
    },
}

# Anchors: Zhang et al., Nature Communications 17, 7466 (2026),
# DOI 10.1038/s41467-026-73610-2, Table 1, PDF pp. 8–9.
# They are diagnostic comparisons, never a strategy assignment or training gate.

PROFILES.update({
    "sh": {
        "title": "Shanghai Metro", "slug": "shanghai", "shape": [288, 6072, 2],
        "stage1": "data/stage1/shanghai.npy", "stage2": "data/stage2/shfine-tuning.npy",
        "checkpoint": "checkpoint/sh/best_model_dict.pth", "interval_minutes": 15,
        "anchors": {"S1": {"MAE": 26.79, "RMSE": 48.49, "WMAPE_percent": 13.48}}},
    "hz": {
        "title": "Hangzhou Metro", "slug": "hangzhou", "shape": [80, 1650, 2],
        "stage1": "data/stage1/hangzhou.npy", "stage2": "data/stage2/hzfine-tuning.npy",
        "checkpoint": "checkpoint/hz/best_model_dict.pth", "interval_minutes": 15,
        "anchors": {"S1": {"MAE": 26.35, "RMSE": 44.15, "WMAPE_percent": 11.43}}},
    "pems03": {
        "title": "PEMS03", "slug": "pems03", "shape": [358, 26208, 2],
        "stage1": "data/stage1/03.npy", "stage2": "data/stage2/03fine-tuning.npy",
        "checkpoint": "checkpoint/03/best_model_dict.pth", "interval_minutes": 5,
        "anchors": {"S1": {"MAE": 10.70, "RMSE": 16.11, "WMAPE_percent": 6.35}}},
    "pems04": {
        "title": "PEMS04", "slug": "pems04", "shape": [307, 16992, 2],
        "stage1": "data/stage1/04.npy", "stage2": "data/stage2/04fine-tunning.npy",
        "checkpoint": "checkpoint/04/best_model_dict.pth", "interval_minutes": 5,
        "anchors": {"S1": {"MAE": 15.99, "RMSE": 25.99, "WMAPE_percent": 7.25}}},
    "bike": {
        "title": "NYC bike", "slug": "nyc_bike", "shape": [64, 8760, 2],
        "stage1": "data/stage1/bike.npy", "stage2": "data/stage2/bikefine-tuning.npy",
        "checkpoint": "checkpoint/bike/best_model_dict.pth", "interval_minutes": 60,
        "anchors": {"S1": {"MAE": 3.67, "RMSE": 6.99, "WMAPE_percent": 19.96}}},
    "taxi": {
        "title": "NYC taxi", "slug": "nyc_taxi", "shape": [64, 8760, 2],
        "stage1": "data/stage1/taxi.npy", "stage2": "data/stage2/taxifine-tuning.npy",
        "checkpoint": "checkpoint/taxi/best_model_dict.pth", "interval_minutes": 60,
        "anchors": {"S1": {"MAE": 16.74, "RMSE": 32.14, "WMAPE_percent": 10.63}}},
})

# Schedule larger road evaluations first. Four subprocesses run at a time.
PUBLIC_ORDER = ("pems03", "pems04", "pems08", "sh", "bj", "hz", "bike", "taxi")
