import csv
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise SystemExit(f"Missing dependency: {exc}. Install required packages first.")

DATA_PATH = Path(__file__).with_name("belyaev_kushnarev.csv")
OUTPUT_DIR = Path(__file__).resolve().parent

FEATURE_COLUMNS = [
    "m1cur",
    "m2cur",
    "m3cur",
    "m1vol",
    "m2vol",
    "m3vol",
    "w1vel",
    "w2vel",
    "w3vel",
]
TARGET_COLUMN = "Ke"


def kalman_1d(z: np.ndarray, q: float = 1e-5, r: float = 1e-2, x0: float | None = None, p0: float = 1.0) -> np.ndarray:
    """Simple 1D Kalman filter (constant model) for a sequence z.

    q: process variance, r: measurement variance
    Returns filtered signal of same shape as z.
    """
    if z is None or len(z) == 0:
        return np.array([])
    z = np.asarray(z, dtype=float)
    n = z.size
    x = np.zeros(n, dtype=float)
    p = p0
    x_prev = x0 if x0 is not None else z[0]
    for i in range(n):
        # prediction (identity)
        x_pred = x_prev
        p_pred = p + q

        # update
        k = p_pred / (p_pred + r)
        x_curr = x_pred + k * (z[i] - x_pred)
        p = (1 - k) * p_pred

        x[i] = x_curr
        x_prev = x_curr

    return x


def load_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def prepare_dataset(rows, limit=None):
    records = []
    for row in rows:
        try:
            features = [float(row[column]) for column in FEATURE_COLUMNS]
            target = float(row[TARGET_COLUMN])
            direction = float(row["movedir"])
            surface = row["surf"]
        except (KeyError, ValueError):
            continue
        records.append((features, target, direction, surface))

    grouped = {}
    for features, target, direction, surface in records:
        grouped.setdefault(surface, []).append((features, target, direction))

    if limit is not None:
        rng = np.random.default_rng(42)
        target_count = min(limit, min(len(items) for items in grouped.values()))
        for surface in list(grouped):
            items = grouped[surface]
            if len(items) > target_count:
                indices = rng.choice(len(items), size=target_count, replace=False)
                grouped[surface] = [items[i] for i in indices]
            else:
                grouped[surface] = items

    datasets = {}
    for surface, items in grouped.items():
        X = np.array([item[0] for item in items], dtype=float)
        y = np.array([item[1] for item in items], dtype=float)
        directions = np.array([item[2] for item in items], dtype=float)
        datasets[surface] = {"X": X, "y": y, "directions": directions}

    return datasets


def train_surface_models(rows, limit=200000):
    datasets = prepare_dataset(rows, limit=limit)
    results = {}

    for surface, data in datasets.items():
        X = data["X"]
        y = data["y"]
        directions = data["directions"]

        if len(X) < 20:
            continue

        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        directions_test = directions[split_idx:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="tanh",
            solver="adam",
            max_iter=200,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=20,
        )
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        error = y_pred - y_test

        # Also predict on the full per-surface dataset so we can summarize by all directions
        X_all = X
        y_all = y
        directions_all = directions
        X_all_scaled = scaler.transform(X_all)
        y_all_pred = model.predict(X_all_scaled)
        # apply Kalman smoothing to the per-surface full predictions so plots and classifier features
        y_all_pred_kf = kalman_1d(y_all_pred, q=1e-5, r=1e-2)
        error_all = y_all_pred - y_all

        mae = float(np.mean(np.abs(error)))
        mse = float(np.mean(error ** 2))
        rmse = float(np.sqrt(mse))

        result = {
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "error": error,
            "directions": directions_test,
            "X_all": X_all,
            "y_all": y_all,
            "y_all_pred": y_all_pred,
            "y_all_pred_kf": y_all_pred_kf,
            "error_all": error_all,
            "directions_all": directions_all,
            "metrics": {"mae": mae, "mse": mse, "rmse": rmse},
            "model": model,
            "scaler": scaler,
            "surface": surface,
        }
        result["direction_summary"] = summarize_by_direction(result)
        results[surface] = result

    return results


def build_surface_classification_dataset(rows, surface_models, limit=None):
    records = []
    for row in rows:
        try:
            features = [float(row[column]) for column in FEATURE_COLUMNS]
            surface = row["surf"]
        except (KeyError, ValueError):
            continue
        records.append((features, surface))

    if limit is not None:
        records = records[:limit]

    feature_rows = []
    labels = []
    surface_names = sorted(surface_models)

    for features, surface in records:
        feature_vector = []
        for name in surface_names:
            model = surface_models[name]["model"]
            scaler = surface_models[name]["scaler"]
            input_array = np.array(features, dtype=float).reshape(1, -1)
            scaled_input = scaler.transform(input_array)
            pred_ke = float(model.predict(scaled_input)[0])
            # apply a short Kalman filter to single-value prediction by treating it as a window of length 1
            # (no-op for single point) — keep original prediction but also allow smoothing at dataset level
            feature_vector.append(pred_ke)
        feature_rows.append(feature_vector)
        labels.append(surface)

    return np.array(labels, dtype=object), np.array(feature_rows, dtype=float)


def train_surface_classifier(rows, surface_models, limit=None):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split

    labels, features = build_surface_classification_dataset(rows, surface_models, limit=limit)
    # Optionally smooth feature columns (per-surface predictions) with Kalman across dataset rows
    try:
        features_kf = np.array([kalman_1d(features[:, i], q=1e-5, r=1e-2) for i in range(features.shape[1])]).T
        # use the smoothed features for classification
        features = features_kf
    except Exception:
        pass
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    classifier = DecisionTreeClassifier(max_depth=4, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    matrix = confusion_matrix(y_test, y_pred, labels=np.unique(labels))

    return {
        "classifier": classifier,
        "classifier_name": classifier.__class__.__name__,
        "accuracy": accuracy,
        "labels": y_test,
        "predicted_labels": y_pred,
        "features": features,
        "confusion_matrix": matrix,
        "classes": np.unique(labels),
    }


def plot_classification_confusion_matrix(classifier_result, output_dir: Path):
    classes = list(classifier_result["classes"])
    matrix = classifier_result["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    ax.set_title("Карта ошибок классификации поверхностей")
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(output_dir / "surface_classifier_confusion_matrix.png", dpi=200)
    plt.close(fig)


def save_results(results, output_dir: Path):
    for surface, result in results.items():
        surface_dir = output_dir / surface
        surface_dir.mkdir(exist_ok=True)

        np.savetxt(surface_dir / "ke_true.csv", result["y_test"], delimiter=",", fmt="%.10f")
        np.savetxt(surface_dir / "ke_pred.csv", result["y_pred"], delimiter=",", fmt="%.10f")
        np.savetxt(surface_dir / "ke_error.csv", result["error"], delimiter=",", fmt="%.10f")

        data = np.column_stack([result["directions"], result["y_test"], result["y_pred"], result["error"]])
        np.savetxt(surface_dir / "ke_comparison.csv", data, delimiter=",", header="direction,true_ke,pred_ke,error", comments="")

        if "direction_summary" in result:
            summary = result["direction_summary"]
            summary_path = surface_dir / "ke_direction_summary.csv"
            with summary_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                header = [
                    "direction",
                    "true_median",
                    "pred_median",
                    "direction_abs_error",
                    "mean_error",
                    "error_rms",
                    "error_rms_25",
                    "error_rms_75",
                    "abs_error_mean",
                    "abs_error_std",
                    "abs_error_25",
                    "abs_error_75",
                    "mean_abs_error",
                ]
                writer.writerow(header)
                for item in summary:
                    writer.writerow([
                        item["direction"],
                        item["true_median"],
                        item["pred_median"],
                        item["direction_abs_error"],
                        item["mean_error"],
                        item["error_rms"],
                        item["error_rms_25"],
                        item["error_rms_75"],
                        item["abs_error_mean"],
                        item["abs_error_std"],
                        item["abs_error_25"],
                        item["abs_error_75"],
                        item["mean_abs_error"],
                    ])

            rms_path = surface_dir / "ke_rmse_by_direction.csv"
            with rms_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["direction", "rmse", "rmse_25", "rmse_75"])
                for item in summary:
                    writer.writerow([
                        item["direction"],
                        item["error_rms"],
                        item["error_rms_25"],
                        item["error_rms_75"],
                    ])

        metrics_path = surface_dir / "model_metrics.csv"
        with metrics_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "value"])
            for name, value in result["metrics"].items():
                writer.writerow([name, value])

        with (surface_dir / "model_metrics.txt").open("w", encoding="utf-8") as handle:
            for name, value in result["metrics"].items():
                handle.write(f"{name}={value:.10f}\n")

    all_summary_rows = []
    rmse_rows = []
    for surface, result in results.items():
        if "direction_summary" not in result:
            continue
        for item in result["direction_summary"]:
            all_summary_rows.append([
                surface,
                item["direction"],
                item["true_median"],
                item["pred_median"],
                item["direction_abs_error"],
                item["mean_error"],
                item["error_rms"],
                item["error_rms_25"],
                item["error_rms_75"],
                item["abs_error_mean"],
                item["abs_error_std"],
                item["abs_error_25"],
                item["abs_error_75"],
                item["mean_abs_error"],
            ])
            rmse_rows.append([
                surface,
                item["direction"],
                item["error_rms"],
                item["error_rms_25"],
                item["error_rms_75"],
            ])

    if all_summary_rows:
        combined_path = output_dir / "ke_direction_summary_all_surfaces.csv"
        with combined_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "surface",
                "direction",
                "true_median",
                "pred_median",
                "direction_abs_error",
                "mean_error",
                "error_rms",
                "error_rms_25",
                "error_rms_75",
                "abs_error_mean",
                "abs_error_std",
                "abs_error_25",
                "abs_error_75",
                "mean_abs_error",
            ])
            writer.writerows(all_summary_rows)

    if rmse_rows:
        rmse_path = output_dir / "ke_rmse_by_direction_all_surfaces.csv"
        with rmse_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["surface", "direction", "rmse", "rmse_25", "rmse_75"])
            writer.writerows(rmse_rows)


def plot_all_points(result, output_dir: Path, surface: str):
    x = np.arange(len(result["y_test"]))
    true_ke = result["y_test"]
    pred_ke = result["y_pred"]
    error = result["error"]

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f"Сравнение исходного Ke, прогнозируемого Ke и ошибки для поверхности {surface}", fontsize=16)

    axes[0].plot(x, true_ke, color="tab:blue", linewidth=1.0, label="Исходное Ke")
    axes[0].set_ylabel("Ke")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, pred_ke, color="tab:orange", linewidth=1.0, label="Прогноз Ke")
    axes[1].set_ylabel("Ke")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(x, error, color="tab:red", linewidth=1.0, label="Ошибка (pred - true)")
    axes[2].set_xlabel("Номер наблюдения")
    axes[2].set_ylabel("Ошибка")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    fig.savefig(output_dir / surface / "ke_comparison_all_points.png", dpi=200)
    plt.close(fig)


def plot_by_direction(result, output_dir: Path, surface: str):
    directions = result["directions"]
    true_ke = result["y_test"]
    pred_ke = result["y_pred"]

    unique_dirs = np.unique(np.round(directions).astype(int))
    medians = []
    preds_by_dir = []
    for direction in unique_dirs:
        mask = np.round(directions).astype(int) == direction
        medians.append((direction, np.median(true_ke[mask])))
        preds_by_dir.append((direction, np.median(pred_ke[mask])))

    dirs_med = [item[0] for item in medians]
    values_med = [item[1] for item in medians]
    dirs_pred = [item[0] for item in preds_by_dir]
    values_pred = [item[1] for item in preds_by_dir]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f"Ke по направлениям движения для поверхности {surface}", fontsize=16)

    axes[0].plot(dirs_med, values_med, marker="o", color="tab:blue", linewidth=1.8, label="Медиана исходного Ke")
    axes[0].set_ylabel("Median Ke")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(dirs_pred, values_pred, marker="o", color="tab:orange", linewidth=1.8, label="Медиана прогнозного Ke")
    axes[1].set_xlabel("Направление движения")
    axes[1].set_ylabel("Predicted Ke")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(output_dir / surface / "ke_by_direction.png", dpi=200)
    plt.close(fig)


def summarize_by_direction(result):
    # Prefer full-dataset predictions if available so summaries cover all directions
    if "directions_all" in result and "y_all_pred" in result:
        directions = result["directions_all"]
        true_ke = result["y_all"]
        pred_ke = result["y_all_pred"]
    else:
        directions = result["directions"]
        true_ke = result["y_test"]
        pred_ke = result["y_pred"]

    unique_dirs = np.unique(np.round(directions).astype(int))
    summary = []
    for direction in sorted(unique_dirs):
        mask = np.round(directions).astype(int) == direction
        true_vals = true_ke[mask]
        pred_vals = pred_ke[mask]
        direction_true_median = float(np.median(true_vals)) if len(true_vals) > 0 else float('nan')
        direction_pred_median = float(np.median(pred_vals)) if len(pred_vals) > 0 else float('nan')
        pred_err = pred_vals - direction_pred_median if len(pred_vals) > 0 else np.array([])
        abs_err = np.abs(pred_vals - true_vals)
        summary.append(
            {
                "direction": int(direction),
                "true_median": direction_true_median,
                "pred_median": direction_pred_median,
                "direction_abs_error": float(abs(direction_pred_median - direction_true_median)) if len(true_vals) > 0 else float('nan'),
                "mean_error": float(np.mean(pred_err)) if len(pred_err) > 0 else float('nan'),
                "error_rms": float(np.sqrt(np.mean(pred_err ** 2))) if len(pred_err) > 0 else float('nan'),
                "error_rms_25": float(np.sqrt(np.percentile(pred_err ** 2, 25))) if len(pred_err) > 0 else float('nan'),
                "error_rms_75": float(np.sqrt(np.percentile(pred_err ** 2, 75))) if len(pred_err) > 0 else float('nan'),
                "abs_error_mean": float(np.mean(abs_err)) if len(abs_err) > 0 else float('nan'),
                "abs_error_std": float(np.std(abs_err)) if len(abs_err) > 0 else float('nan'),
                "abs_error_25": float(np.percentile(abs_err, 25)) if len(abs_err) > 0 else float('nan'),
                "abs_error_75": float(np.percentile(abs_err, 75)) if len(abs_err) > 0 else float('nan'),
                "mean_abs_error": float(np.mean(abs_err)) if len(abs_err) > 0 else float('nan'),
            }
        )
    return summary


def plot_direction_comparison(result, output_dir: Path, surface: str):
    summary = summarize_by_direction(result)
    x = [item["direction"] for item in summary]
    pred_medians = [item["pred_median"] for item in summary]
    rms_errors = [item["error_rms"] for item in summary]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(x, pred_medians, marker="s", color="tab:red", linewidth=2, label="Медиана прогноза Ke")

    lower = np.array(pred_medians) - np.array(rms_errors)
    upper = np.array(pred_medians) + np.array(rms_errors)
    ax.fill_between(x, lower, upper, color="tab:red", alpha=0.2, label="Возможный диапазон прогноза Ke (±RMSE)")
    ax.plot(x, lower, color="tab:red", linewidth=1.0, linestyle="--", alpha=0.7, label="Нижняя граница прогноза")
    ax.plot(x, upper, color="tab:red", linewidth=1.0, linestyle="--", alpha=0.7, label="Верхняя граница прогноза")

    ax.set_title(f"Возможные значения Ke в прогнозе нейронной сети для поверхности {surface}")
    ax.set_xlabel("Направление движения")
    ax.set_ylabel("Ke")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / surface / "ke_direction_comparison_with_rms.png", dpi=200)
    plt.close(fig)


def plot_direction_comparison_dual_axis(result, output_dir: Path, surface: str):
    summary = summarize_by_direction(result)
    x = [item["direction"] for item in summary]
    true_medians = [item["true_median"] for item in summary]
    pred_medians = [item["pred_median"] for item in summary]
    direction_abs_errors = [item["direction_abs_error"] for item in summary]

    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax1.plot(x, true_medians, marker="o", color="tab:blue", linewidth=2, label="Исходное Ke")
    ax1.plot(x, pred_medians, marker="s", color="tab:red", linewidth=2, label="Прогноз Ke")
    ax1.set_xlabel("Направление движения")
    ax1.set_ylabel("Ke", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, direction_abs_errors, marker="d", color="tab:purple", linewidth=2, linestyle="--", label="Ошибка по медианам")
    ax2.set_ylabel("Ошибка Ke", color="tab:purple")
    ax2.tick_params(axis="y", labelcolor="tab:purple")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title(f"Ke и ошибка по направлениям для поверхности {surface} (две оси)")
    plt.tight_layout()
    fig.savefig(output_dir / surface / "ke_direction_comparison_dual_axis.png", dpi=200)
    plt.close(fig)


def plot_combined_predictions(results, output_dir: Path):
    fig, ax = plt.subplots(figsize=(16, 8))
    colors = {"gray": "tab:blue", "green": "tab:green", "red": "tab:red", "table": "tab:purple"}

    for surface, result in results.items():
        summary = summarize_by_direction(result)
        obs_dirs = np.array([item["direction"] for item in summary], dtype=float)
        obs_pred = np.array([item["pred_median"] for item in summary], dtype=float)
        obs_err = np.array([item["direction_abs_error"] for item in summary], dtype=float)

        if len(obs_dirs) == 0:
            continue

        err_min = np.min(obs_err) if obs_err.size > 0 else 0.0
        err_max = np.max(obs_err) if obs_err.size > 0 else 0.0
        if err_max > err_min:
            normalized_err = (obs_err - err_min) / (err_max - err_min)
        else:
            normalized_err = np.zeros_like(obs_err)

        pred_min, pred_max = obs_pred.min(), obs_pred.max()
        scaled_err = normalized_err * (pred_max - pred_min) * 0.2
        lower = obs_pred - scaled_err
        upper = obs_pred + scaled_err
        color = colors.get(surface, "tab:gray")
        ax.plot(obs_dirs, obs_pred, marker="o", markersize=4, linewidth=2, label=f"{surface} (pred)", color=color)
        ax.fill_between(obs_dirs, lower, upper, color=color, alpha=0.2, label=f"{surface} нормированная ошибка")

    ax.set_title("Прогноз Ke по направлениям для всех поверхностей")
    ax.set_xlabel("Направление движения")
    ax.set_ylabel("Прогноз Ke")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    plt.tight_layout()
    fig.savefig(output_dir / "ke_predictions_all_surfaces.png", dpi=200)
    plt.close(fig)


def plot_combined_predictions_scaled_error(results, output_dir: Path):
    fig, ax1 = plt.subplots(figsize=(16, 8))
    colors = {"gray": "tab:blue", "green": "tab:green", "red": "tab:red", "table": "tab:purple"}

    for surface, result in results.items():
        summary = summarize_by_direction(result)
        obs_dirs = np.array([item["direction"] for item in summary], dtype=float)
        obs_pred = np.array([item["pred_median"] for item in summary], dtype=float)
        obs_err = np.array([item["direction_abs_error"] for item in summary], dtype=float)

        if len(obs_dirs) == 0:
            continue

        pred_min, pred_max = obs_pred.min(), obs_pred.max()
        if obs_err.max() > 0:
            scaled_err = (obs_err / obs_err.max()) * (pred_max - pred_min) * 0.3 + pred_min
        else:
            scaled_err = np.full_like(obs_err, pred_min)

        color = colors.get(surface, "tab:gray")
        ax1.plot(obs_dirs, obs_pred, marker="o", markersize=4, linewidth=2, label=f"{surface} (pred)", color=color)
        ax1.fill_between(obs_dirs, obs_pred - scaled_err, obs_pred + scaled_err, color=color, alpha=0.25)

    ax1.set_title("Прогноз Ke и масштабированная ошибка по направлениям для всех поверхностей")
    ax1.set_xlabel("Направление движения")
    ax1.set_ylabel("Ke / Масштабированная ошибка")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", ncol=2)

    plt.tight_layout()
    fig.savefig(output_dir / "ke_predictions_all_surfaces_scaled_error.png", dpi=200)
    plt.close(fig)


def plot_combined_direction_error_percentiles(results, output_dir: Path):
    grouped = {}
    for result in results.values():
        if "directions_all" not in result or "error_all" not in result:
            continue
        for direction, err in zip(result["directions_all"], result["error_all"]):
            direction_key = int(np.round(direction))
            grouped.setdefault(direction_key, []).append(abs(err))

    if not grouped:
        return

    directions = sorted(grouped)
    error_25 = [np.percentile(grouped[d], 25) for d in directions]
    error_75 = [np.percentile(grouped[d], 75) for d in directions]
    error_mean = [np.mean(grouped[d]) for d in directions]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(directions, error_mean, marker="o", linestyle="-", color="tab:blue", linewidth=2, label="Средняя абсолютная ошибка")
    ax.fill_between(directions, error_25, error_75, color="tab:blue", alpha=0.2, label="25-75 процентиль абсолютной ошибки")

    ax.set_title("По-направлениям распределение абсолютной ошибки Ke для всех поверхностей")
    ax.set_xlabel("Направление движения")
    ax.set_ylabel("Абсолютная ошибка Ke")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "ke_direction_error_percentiles_all_surfaces.png", dpi=200)
    plt.close(fig)


def plot_combined_direction_error_percentiles_by_surface(results, output_dir: Path):
    colors = {"gray": "tab:blue", "green": "tab:green", "red": "tab:red", "table": "tab:purple"}
    fig, ax = plt.subplots(figsize=(16, 8))

    for surface, result in results.items():
        summary = summarize_by_direction(result)
        x = np.array([item["direction"] for item in summary], dtype=float)
        error_25 = np.array([item["abs_error_25"] for item in summary], dtype=float)
        error_75 = np.array([item["abs_error_75"] for item in summary], dtype=float)
        error_mean = np.array([item["abs_error_mean"] for item in summary], dtype=float)

        if len(x) == 0:
            continue

        color = colors.get(surface, "tab:gray")
        ax.plot(x, error_mean, marker="o", markersize=3, linewidth=1.5, linestyle="-", color=color, label=f"{surface} mean abs error")
        ax.fill_between(x, error_25, error_75, color=color, alpha=0.15, label=f"{surface} 25-75%")

    ax.set_title("25-75 процентиль абсолютной ошибки по направлениям для всех поверхностей")
    ax.set_xlabel("Направление движения")
    ax.set_ylabel("Абсолютная ошибка Ke")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    plt.tight_layout()
    fig.savefig(output_dir / "ke_direction_error_percentiles_by_surface.png", dpi=200)
    plt.close(fig)


def plot_combined_direction_rms_and_percentiles_by_surface(results, output_dir: Path):
    colors = {"gray": "tab:blue", "green": "tab:green", "red": "tab:red", "table": "tab:purple"}
    fig, ax = plt.subplots(figsize=(16, 8))

    min_direction, max_direction = 0, 360
    band_width = 30
    for start in range(min_direction, max_direction, band_width * 2):
        ax.axvspan(start, min(start + band_width, max_direction), color="gray", alpha=0.08)

    for surface, result in results.items():
        summary = summarize_by_direction(result)
        x = np.array([item["direction"] for item in summary], dtype=float)
        pred_medians = np.array([item["pred_median"] for item in summary], dtype=float)
        rms_25 = np.array([item["error_rms_25"] for item in summary], dtype=float)
        rms_75 = np.array([item["error_rms_75"] for item in summary], dtype=float)

        if len(x) == 0:
            continue

        lower_75 = pred_medians - rms_75
        upper_75 = pred_medians + rms_75
        lower_25 = pred_medians - rms_25
        upper_25 = pred_medians + rms_25

        color = colors.get(surface, "tab:gray")
        ax.fill_between(x, lower_75, upper_75, color=color, alpha=0.12, label=f"{surface} 75% RMSE", zorder=1)
        ax.fill_between(x, lower_25, upper_25, color=color, alpha=0.22, label=f"{surface} 25% RMSE", zorder=2)
        ax.plot(x, lower_75, color=color, linewidth=0.8, linestyle="--", alpha=0.5, zorder=3)
        ax.plot(x, upper_75, color=color, linewidth=0.8, linestyle="--", alpha=0.5, zorder=3)
        ax.plot(x, lower_25, color=color, linewidth=0.8, linestyle=":", alpha=0.5, zorder=3)
        ax.plot(x, upper_25, color=color, linewidth=0.8, linestyle=":", alpha=0.5, zorder=3)
        ax.plot(x, pred_medians, marker="o", markersize=3, linewidth=1.5, linestyle="-", color=color, label=f"{surface} pred median", zorder=4)

    ax.set_title("Средние прогнозы Ke и RMSE-диапазоны по направлениям для всех поверхностей")
    ax.set_xlabel("Направление движения")
    ax.set_ylabel("Ke / RMSE диапазон")
    ax.set_xlim(min_direction, max_direction)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    plt.tight_layout()
    fig.savefig(output_dir / "ke_direction_rms_and_percentiles_by_surface.png", dpi=200)
    plt.close(fig)


def plot_combined_direction_comparison_with_rms(results, output_dir: Path):
    colors = {"gray": "tab:blue", "green": "tab:green", "red": "tab:red", "table": "tab:purple"}
    fig, ax = plt.subplots(figsize=(16, 8))

    for surface, result in results.items():
        # prefer Kalman-smoothed full predictions when available
        if "y_all_pred_kf" in result:
            preds = result["y_all_pred_kf"]
        else:
            preds = result.get("y_all_pred", np.array([]))

        directions = result.get("directions_all", result.get("directions", np.array([])))
        if preds.size == 0 or directions.size == 0:
            continue

        # summarize per-direction using the (smoothed) predictions only
        uniq_dirs = np.unique(np.round(directions).astype(int))
        x = []
        pred_medians = []
        rms_errors = []
        for d in sorted(uniq_dirs):
            mask = np.round(directions).astype(int) == d
            vals = preds[mask]
            if vals.size == 0:
                continue
            x.append(int(d))
            median = float(np.median(vals))
            pred_medians.append(median)
            pred_err = vals - median
            rms = float(np.sqrt(np.mean(pred_err ** 2))) if vals.size > 0 else float('nan')
            rms_errors.append(rms)

        if len(x) == 0:
            continue

        color = colors.get(surface, "tab:gray")
        ax.plot(x, pred_medians, marker="s", linewidth=1.8, linestyle="-", color=color, label=f"{surface} pred median")
        ax.fill_between(x, np.array(pred_medians) - np.array(rms_errors), np.array(pred_medians) + np.array(rms_errors), color=color, alpha=0.15, zorder=1, label=f"{surface} диапазон прогноза (±RMSE)")
        ax.plot(x, np.array(pred_medians) - np.array(rms_errors), color=color, linewidth=1.0, linestyle="--", alpha=0.5)
        ax.plot(x, np.array(pred_medians) + np.array(rms_errors), color=color, linewidth=1.0, linestyle="--", alpha=0.5)

    ax.set_title("Общий прогнозный диапазон Ke по направлениям для всех поверхностей")
    ax.set_xlabel("Направление движения")
    ax.set_ylabel("Ke")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    plt.tight_layout()
    fig.savefig(output_dir / "ke_direction_comparison_with_rms_all_surfaces.png", dpi=200)
    plt.close(fig)


def plot_direction_error_comparison(result, output_dir: Path, surface: str):
    summary = summarize_by_direction(result)
    x = [item["direction"] for item in summary]
    direction_abs_errors = [item["direction_abs_error"] for item in summary]
    abs_error_means = [item["abs_error_mean"] for item in summary]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(x, direction_abs_errors, marker="o", color="tab:red", linewidth=2, label="|Медиана pred - медиана true|")
    ax.plot(x, abs_error_means, marker="s", color="tab:orange", linewidth=2, label="Средняя абсолютная ошибка по точкам")

    ax.set_title(f"Сравнение ошибок по направлениям для поверхности {surface}")
    ax.set_xlabel("Направление движения")
    ax.set_ylabel("Ошибка Ke")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / surface / "ke_direction_error_comparison.png", dpi=200)
    plt.close(fig)


def plot_direction_error_percentiles(result, output_dir: Path, surface: str):
    summary = summarize_by_direction(result)
    x = [item["direction"] for item in summary]
    abs_error_25 = [item["abs_error_25"] for item in summary]
    abs_error_75 = [item["abs_error_75"] for item in summary]
    abs_error_means = [item["abs_error_mean"] for item in summary]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(x, abs_error_means, marker="o", color="tab:blue", linewidth=2, label="Средняя абсолютная ошибка")
    ax.fill_between(x, abs_error_25, abs_error_75, color="tab:blue", alpha=0.2, label="25-75 процентиль по абсолютной ошибке")

    ax.set_title(f"Percentile ошибок по направлениям для поверхности {surface}")
    ax.set_xlabel("Направление движения")
    ax.set_ylabel("Абсолютная ошибка Ke")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / surface / "ke_direction_error_percentiles.png", dpi=200)
    plt.close(fig)


def plot_direction_rms_error(result, output_dir: Path, surface: str):
    summary = summarize_by_direction(result)
    x = [item["direction"] for item in summary]
    rms_errors = [item["error_rms"] for item in summary]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(x, rms_errors, marker="o", color="tab:purple", linewidth=2, label="RMSE по направлению")

    ax.set_title(f"Среднеквадратичная ошибка Ke по направлениям для поверхности {surface}")
    ax.set_xlabel("Направление движения")
    ax.set_ylabel("RMSE Ke")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / surface / "ke_direction_rms_error.png", dpi=200)
    plt.close(fig)


def main():
    rows = load_rows(DATA_PATH)
    surface_models = train_surface_models(rows, limit=None)
    save_results(surface_models, OUTPUT_DIR)

    for surface, result in surface_models.items():
        plot_all_points(result, OUTPUT_DIR, surface)
        plot_by_direction(result, OUTPUT_DIR, surface)
        plot_direction_comparison(result, OUTPUT_DIR, surface)
        plot_direction_comparison_dual_axis(result, OUTPUT_DIR, surface)
        plot_direction_error_comparison(result, OUTPUT_DIR, surface)
        plot_direction_error_percentiles(result, OUTPUT_DIR, surface)
        plot_direction_rms_error(result, OUTPUT_DIR, surface)
    plot_combined_predictions(surface_models, OUTPUT_DIR)
    plot_combined_predictions_scaled_error(surface_models, OUTPUT_DIR)
    plot_combined_direction_error_percentiles(surface_models, OUTPUT_DIR)
    plot_combined_direction_error_percentiles_by_surface(surface_models, OUTPUT_DIR)
    plot_combined_direction_rms_and_percentiles_by_surface(surface_models, OUTPUT_DIR)
    plot_combined_direction_comparison_with_rms(surface_models, OUTPUT_DIR)

    classifier_result = train_surface_classifier(rows, surface_models, limit=None)
    plot_classification_confusion_matrix(classifier_result, OUTPUT_DIR)
    classifier_path = OUTPUT_DIR / "surface_classifier_metrics.csv"
    with classifier_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["classifier_name", classifier_result["classifier_name"]])
        writer.writerow(["accuracy", classifier_result["accuracy"]])

    labels_path = OUTPUT_DIR / "surface_classifier_labels.csv"
    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_surface", "predicted_surface"])
        for true_label, predicted_label in zip(classifier_result["labels"], classifier_result["predicted_labels"]):
            writer.writerow([true_label, predicted_label])

    confusion_path = OUTPUT_DIR / "surface_classifier_confusion_matrix.csv"
    with confusion_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", *classifier_result["classes"]])
        for idx, class_name in enumerate(classifier_result["classes"]):
            writer.writerow([class_name, *classifier_result["confusion_matrix"][idx]])

    print("Loaded rows:", len(rows))
    for surface, result in surface_models.items():
        print(surface, result["metrics"])
    print("Surface classifier accuracy:", classifier_result["accuracy"])
    print("Confusion matrix:")
    print(classifier_result["confusion_matrix"])
    print("Saved surface folders:")
    for surface in surface_models:
        print("-", OUTPUT_DIR / surface)


if __name__ == "__main__":
    main()
