#!/usr/bin/env python3
import csv
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBRegressor


MYSQL = os.environ.get("MYSQL_BIN", "/Applications/MAMP/Library/bin/mysql80/bin/mysql")
DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
DB_USER = os.environ.get("DB_USER", "root")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "root")
DB_NAME = os.environ.get("DB_NAME", "visuelle")
SEED = int(os.environ.get("MODEL_SEED", "580"))
HORIZONS = [4, 8, 12]
OUT_REPORT = Path("public/data/xgboost-model-validation.json")
OUT_EXISTING_ARTIFACT = Path("public/data/existing-demand-model.json")
OUT_NEW_ARTIFACT = Path("public/data/new-product-demand-model.json")


def mysql_tsv(sql, header=False):
    args = [
        MYSQL,
        f"-h{DB_HOST}",
        f"-u{DB_USER}",
        f"-p{DB_PASSWORD}",
        "--batch",
        "--raw",
    ]
    if not header:
        args.append("--skip-column-names")
    args.extend([DB_NAME, "-e", sql])
    return subprocess.check_output(args, text=True)


def parse_tsv(text, fieldnames=None):
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    reader = csv.DictReader(lines, delimiter="\t", fieldnames=fieldnames)
    return list(reader)


def to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except ValueError:
        return default


def slug(value):
    return "".join(ch if ch.isalnum() else "_" for ch in str(value).strip().lower()).strip("_")


def combo_key(category, color, fabric):
    return f"{category}||{color}||{fabric}"


def stable_hash(value):
    h = 2166136261 ^ SEED
    for ch in str(value):
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def split_ids(ids):
    ordered = sorted(ids, key=stable_hash)
    train_end = int(len(ordered) * 0.7)
    val_end = train_end + int(len(ordered) * 0.15)
    result = {}
    for idx, product_id in enumerate(ordered):
        result[product_id] = "train" if idx < train_end else "validation" if idx < val_end else "test"
    return result


def metrics(rows, predictions, target_key):
    actual = np.array([row[target_key] for row in rows], dtype=float)
    forecast = np.maximum(0, np.array(predictions, dtype=float))
    error = forecast - actual
    abs_error = np.abs(error)
    actual_total = max(1.0, float(actual.sum()))
    wape = float(abs_error.sum() / actual_total)
    return {
        "samples": len(rows),
        "products": len({row["external_code"] for row in rows}),
        "actualUnits": int(round(actual.sum())),
        "forecastUnits": int(round(forecast.sum())),
        "mae": round(float(abs_error.mean()), 2),
        "rmse": round(float(np.sqrt(np.mean(error ** 2))), 2),
        "wape": round(wape, 4),
        "wapePct": round(wape * 100, 2),
        "accuracyPct": round(max(0.0, 1 - wape) * 100, 2),
        "bias": round(float(error.sum() / actual_total), 4),
        "biasPct": round(float(error.sum() / actual_total) * 100, 2),
    }


def train_xgb(train_rows, val_rows, test_rows, feature_fn, target_key):
    param_grid = [
        {"max_depth": 2, "n_estimators": 100, "learning_rate": 0.06, "subsample": 0.9, "colsample_bytree": 0.9},
        {"max_depth": 3, "n_estimators": 120, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9},
        {"max_depth": 3, "n_estimators": 180, "learning_rate": 0.04, "subsample": 0.9, "colsample_bytree": 0.9},
        {"max_depth": 4, "n_estimators": 140, "learning_rate": 0.04, "subsample": 0.85, "colsample_bytree": 0.85},
    ]
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform([feature_fn(row) for row in train_rows])
    x_val = vectorizer.transform([feature_fn(row) for row in val_rows])
    x_test = vectorizer.transform([feature_fn(row) for row in test_rows])
    y_train = np.log1p([row[target_key] for row in train_rows])

    candidates = []
    for params in param_grid:
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=4,
            reg_lambda=1.0,
            min_child_weight=3,
            **params,
        )
        model.fit(x_train, y_train)
        val_pred = np.expm1(model.predict(x_val))
        candidates.append({
            "model": model,
            "params": params,
            "validation": metrics(val_rows, val_pred, target_key),
        })

    best = min(candidates, key=lambda item: item["validation"]["wape"])
    model = best["model"]
    train_pred = np.expm1(model.predict(x_train))
    val_pred = np.expm1(model.predict(x_val))
    test_pred = np.expm1(model.predict(x_test))
    return {
        "model": model,
        "vectorizer": vectorizer,
        "params": best["params"],
        "split": {
            "train": metrics(train_rows, train_pred, target_key),
            "validation": metrics(val_rows, val_pred, target_key),
            "test": metrics(test_rows, test_pred, target_key),
        },
        "validationTuning": [
            {"params": item["params"], **item["validation"]}
            for item in sorted(candidates, key=lambda item: item["validation"]["wape"])
        ],
    }


def load_base_rows():
    sql = """
    SELECT external_code, COALESCE(category,'' ) category, COALESCE(color,'' ) color,
           COALESCE(fabric,'' ) fabric, COALESCE(release_date,'1970-01-01') release_date,
           week_since_release, units
    FROM product_weekly_lifecycle
    WHERE external_code IS NOT NULL
      AND category IS NOT NULL
      AND color IS NOT NULL
      AND fabric IS NOT NULL
      AND week_since_release IS NOT NULL
      AND week_since_release BETWEEN 0 AND 23
      AND units IS NOT NULL
    ORDER BY external_code, week_since_release;
    """
    rows = parse_tsv(mysql_tsv(sql), ["external_code", "category", "color", "fabric", "release_date", "week", "units"])
    by_product = {}
    for row in rows:
        product_id = row["external_code"]
        product = by_product.setdefault(product_id, {
            "external_code": product_id,
            "category": row["category"],
            "color": row["color"],
            "fabric": row["fabric"],
            "release_date": row["release_date"],
            "weekly": {},
        })
        product["weekly"][int(float(row["week"]))] = to_float(row["units"])
    return by_product


def load_price_map():
    sql = """
    SELECT external_code,
           AVG(price) avg_price,
           AVG((COALESCE(week_0,0)+COALESCE(week_1,0)+COALESCE(week_2,0)+COALESCE(week_3,0))/4) avg_discount_w0_w3,
           MAX(GREATEST(COALESCE(week_0,0),COALESCE(week_1,0),COALESCE(week_2,0),COALESCE(week_3,0))) max_discount_w0_w3
    FROM price_discount_series
    WHERE external_code IS NOT NULL
    GROUP BY external_code;
    """
    result = {}
    for row in parse_tsv(mysql_tsv(sql), ["external_code", "avg_price", "avg_discount_w0_w3", "max_discount_w0_w3"]):
        result[row["external_code"]] = {key: to_float(row[key]) for key in row if key != "external_code"}
    return result


def load_store_map():
    sql = """
    SELECT external_code,
           COUNT(DISTINCT retail) store_coverage,
           SUM(COALESCE(restock,0)) restock_signal,
           SUM(CASE WHEN COALESCE(restock,0)>0 THEN 1 ELSE 0 END) restock_store_rows
    FROM sales
    WHERE external_code IS NOT NULL
    GROUP BY external_code;
    """
    result = {}
    for row in parse_tsv(mysql_tsv(sql), ["external_code", "store_coverage", "restock_signal", "restock_store_rows"]):
        result[row["external_code"]] = {key: to_float(row[key]) for key in row if key != "external_code"}
    return result


def load_trends():
    rows = parse_tsv(mysql_tsv("SELECT * FROM vis2_gtrends_data ORDER BY date;", header=True))
    return rows


def trend_row_for(trend_rows, release_date, week_offset=0):
    if not trend_rows:
        return {}
    try:
        target = datetime.strptime(release_date, "%Y-%m-%d") + timedelta(days=7 * week_offset)
    except ValueError:
        return trend_rows[0]
    return min(trend_rows, key=lambda row: abs((datetime.strptime(row["date"], "%Y-%m-%d") - target).days))


def trend_features(product, trend_rows, cutoff_week=0):
    row = trend_row_for(trend_rows, product.get("release_date", "1970-01-01"), cutoff_week)
    category = to_float(row.get(slug(product.get("category"))))
    color = to_float(row.get(slug(product.get("color"))))
    fabric = to_float(row.get(slug(product.get("fabric"))))
    return {
        "category_trend": category / 100,
        "color_trend": color / 100,
        "fabric_trend": fabric / 100,
        "combined_trend": ((category * 0.5) + (color * 0.25) + (fabric * 0.25)) / 100,
    }


def build_group_means(rows, target_key):
    groups = ["category", "color", "fabric", "category_color", "category_fabric", "color_fabric", "category_color_fabric"]
    stats = {group: defaultdict(lambda: [0.0, 0]) for group in groups}
    global_mean = sum(row[target_key] for row in rows) / max(1, len(rows))

    for row in rows:
        keys = {
            "category": row["category"],
            "color": row["color"],
            "fabric": row["fabric"],
            "category_color": f"{row['category']}||{row['color']}",
            "category_fabric": f"{row['category']}||{row['fabric']}",
            "color_fabric": f"{row['color']}||{row['fabric']}",
            "category_color_fabric": f"{row['category']}||{row['color']}||{row['fabric']}",
        }
        for group, key in keys.items():
            stats[group][key][0] += row[target_key]
            stats[group][key][1] += 1
    return {"global_mean": global_mean, "stats": stats}


def group_features(row, group_model):
    features = {}
    keys = {
        "category": row["category"],
        "color": row["color"],
        "fabric": row["fabric"],
        "category_color": f"{row['category']}||{row['color']}",
        "category_fabric": f"{row['category']}||{row['fabric']}",
        "color_fabric": f"{row['color']}||{row['fabric']}",
        "category_color_fabric": f"{row['category']}||{row['color']}||{row['fabric']}",
    }
    global_mean = group_model["global_mean"]
    prior = 20
    for group, key in keys.items():
        total, count = group_model["stats"][group].get(key, [0.0, 0])
        smoothed = (total + global_mean * prior) / (count + prior)
        features[f"{group}_benchmark_ratio"] = smoothed / max(1, global_mean)
        features[f"{group}_count_log"] = math.log1p(count)
    return features


def make_new_product_rows(products):
    rows = []
    for product in products.values():
        weekly = product["weekly"]
        if all(week in weekly for week in range(12)):
            row = {key: product[key] for key in ["external_code", "category", "color", "fabric", "release_date"]}
            for horizon in HORIZONS:
                row[f"future{horizon}wDemand"] = sum(weekly[w] for w in range(horizon))
            rows.append(row)
    return rows


def make_existing_rows(products, price_map, store_map, trend_rows):
    rows = []
    for product in products.values():
        weekly = product["weekly"]
        max_week = max(weekly) if weekly else -1
        for cutoff in range(3, max_week - 12 + 1):
            if not all(w in weekly for w in range(cutoff - 3, cutoff + 13)):
                continue
            recent = [weekly[w] for w in range(cutoff - 3, cutoff + 1)]
            first2 = recent[0] + recent[1]
            last2 = recent[2] + recent[3]
            total = sum(recent)
            row = {key: product[key] for key in ["external_code", "category", "color", "fabric", "release_date"]}
            row.update({
                "cutoff_week": cutoff,
                "observed4w_units": total,
                "observed_first2w_units": first2,
                "observed_last2w_units": last2,
                "observed_trend_ratio": last2 / first2 if first2 > 0 else 1,
                "week0_share": recent[0] / max(1, total),
                "week1_share": recent[1] / max(1, total),
                "week2_share": recent[2] / max(1, total),
                "week3_share": recent[3] / max(1, total),
            })
            row.update(price_map.get(product["external_code"], {}))
            row.update(store_map.get(product["external_code"], {}))
            row.update(trend_features(product, trend_rows, cutoff))
            for key in ["avg_price", "avg_discount_w0_w3", "max_discount_w0_w3", "store_coverage", "restock_signal", "restock_store_rows"]:
                row.setdefault(key, 0)
            for horizon in HORIZONS:
                row[f"next{horizon}wDemand"] = sum(weekly[w] for w in range(cutoff + 1, cutoff + horizon + 1))
            rows.append(row)
    return rows


def base_new_features(row, group_model=None):
    features = {
        "category": row["category"],
        "color": row["color"],
        "fabric": row["fabric"],
    }
    if group_model:
        features.update(group_features(row, group_model))
    return features


def base_existing_features(row, group_model=None):
    features = {
        "category": row["category"],
        "color": row["color"],
        "fabric": row["fabric"],
        "log_observed4w_units": math.log1p(row.get("observed4w_units", 0)),
        "log_observed_first2w_units": math.log1p(row.get("observed_first2w_units", 0)),
        "log_observed_last2w_units": math.log1p(row.get("observed_last2w_units", 0)),
        "observed_trend_ratio_capped": max(0, min(5, row.get("observed_trend_ratio", 1))),
        "week0_share": row.get("week0_share", 0),
        "week1_share": row.get("week1_share", 0),
        "week2_share": row.get("week2_share", 0),
        "week3_share": row.get("week3_share", 0),
        "avg_price": row.get("avg_price", 0),
        "avg_discount_w0_w3": row.get("avg_discount_w0_w3", 0),
        "max_discount_w0_w3": row.get("max_discount_w0_w3", 0),
        "log_store_coverage": math.log1p(row.get("store_coverage", 0)),
        "log_restock_signal": math.log1p(row.get("restock_signal", 0)),
        "log_restock_store_rows": math.log1p(row.get("restock_store_rows", 0)),
        "category_trend": row.get("category_trend", 0),
        "color_trend": row.get("color_trend", 0),
        "fabric_trend": row.get("fabric_trend", 0),
        "combined_trend": row.get("combined_trend", 0),
    }
    if group_model:
        features.update(group_features(row, group_model))
    return features


def train_family(rows, target_prefix, feature_builder):
    split_map = split_ids(sorted({row["external_code"] for row in rows}))
    for row in rows:
        row["split"] = split_map[row["external_code"]]
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "validation"]
    test_rows = [row for row in rows if row["split"] == "test"]
    result = {
        "splitPolicy": {
            "method": "Product-level deterministic 70/15/15 split by external_code hash",
            "trainProducts": len({row["external_code"] for row in train_rows}),
            "validationProducts": len({row["external_code"] for row in val_rows}),
            "testProducts": len({row["external_code"] for row in test_rows}),
            "trainSamples": len(train_rows),
            "validationSamples": len(val_rows),
            "testSamples": len(test_rows),
        },
        "horizons": {},
    }
    trained = {}
    for horizon in HORIZONS:
        target = f"{target_prefix}{horizon}wDemand"
        group_model = build_group_means(train_rows, target)
        trained_model = train_xgb(
            train_rows,
            val_rows,
            test_rows,
            lambda row, gm=group_model: feature_builder(row, gm),
            target,
        )
        result["horizons"][str(horizon)] = {
            "horizonWeeks": horizon,
            "selectedModel": {
                "type": "XGBoost Regressor",
                "targetMode": "log1p",
                "params": trained_model["params"],
            },
            "split": trained_model["split"],
            "validationTuning": trained_model["validationTuning"],
        }
        trained[str(horizon)] = {
            **trained_model,
            "group_model": group_model,
            "target": target,
        }
    return result, trained


def demo_existing_predictions(demo_products, trained_existing, price_map, store_map, trend_rows):
    predictions = {}
    for product in demo_products:
        actual = product.get("actualLifecycle") or product.get("weekSales") or []
        if len(actual) < 4:
            continue
        recent = actual[-4:]
        first2 = recent[0] + recent[1]
        last2 = recent[2] + recent[3]
        total = sum(recent)
        row = {
            "external_code": str(product["id"]),
            "category": product["category"],
            "color": product["color"],
            "fabric": product["fabric"],
            "release_date": product.get("releaseDate", "1970-01-01"),
            "cutoff_week": len(actual) - 1,
            "observed4w_units": total,
            "observed_first2w_units": first2,
            "observed_last2w_units": last2,
            "observed_trend_ratio": last2 / first2 if first2 > 0 else 1,
            "week0_share": recent[0] / max(1, total),
            "week1_share": recent[1] / max(1, total),
            "week2_share": recent[2] / max(1, total),
            "week3_share": recent[3] / max(1, total),
            "avg_price": product.get("price", 0),
            "avg_discount_w0_w3": product.get("avgDiscount", 0),
            "max_discount_w0_w3": product.get("avgDiscount", 0),
            "store_coverage": product.get("stores", 0),
            "restock_signal": product.get("restockEvents", 0),
            "restock_store_rows": product.get("restockEvents", 0),
        }
        trend = product.get("googleTrendSignal", {})
        trend_value = trend.get("forecastAvg", trend.get("recentAvg", 0)) or 0
        row.update({
            "category_trend": trend_value / 100,
            "color_trend": trend_value / 100,
            "fabric_trend": trend_value / 100,
            "combined_trend": trend_value / 100,
        })
        predictions[str(product["id"])] = {}
        for horizon, model_pack in trained_existing.items():
            features = [base_existing_features(row, model_pack["group_model"])]
            x = model_pack["vectorizer"].transform(features)
            prediction = float(np.expm1(model_pack["model"].predict(x))[0])
            predictions[str(product["id"])][f"next{horizon}wDemand"] = max(0, round(prediction))
    return predictions


def new_product_predictions(attribute_options, trained_new):
    predictions = {}
    for category in attribute_options.get("categories", []):
        for color in attribute_options.get("colors", []):
            for fabric in attribute_options.get("fabrics", []):
                row = {
                    "external_code": combo_key(category, color, fabric),
                    "category": category,
                    "color": color,
                    "fabric": fabric,
                    "release_date": "1970-01-01",
                }
                key = combo_key(category, color, fabric)
                predictions[key] = {}
                for horizon, model_pack in trained_new.items():
                    features = [base_new_features(row, model_pack["group_model"])]
                    x = model_pack["vectorizer"].transform(features)
                    prediction = float(np.expm1(model_pack["model"].predict(x))[0])
                    predictions[key][f"future{horizon}wDemand"] = max(0, round(prediction))
    return predictions


def category_launch_benchmarks(rows):
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row["future12wDemand"])
    benchmarks = {}
    for category, values in by_category.items():
        sorted_values = sorted(values)
        if not sorted_values:
            continue
        benchmarks[category] = {
            "median": round(percentile(sorted_values, 0.5)),
            "topQuartile": round(percentile(sorted_values, 0.75)),
            "count": len(sorted_values),
        }
    return benchmarks


def percentile(sorted_values, q):
    if not sorted_values:
        return 0
    position = (len(sorted_values) - 1) * q
    base = math.floor(position)
    rest = position - base
    if base + 1 >= len(sorted_values):
        return sorted_values[base]
    return sorted_values[base] + rest * (sorted_values[base + 1] - sorted_values[base])


def main():
    products = load_base_rows()
    price_map = load_price_map()
    store_map = load_store_map()
    trend_rows = load_trends()
    new_rows = make_new_product_rows(products)
    existing_rows = make_existing_rows(products, price_map, store_map, trend_rows)

    new_report, trained_new = train_family(new_rows, "future", base_new_features)
    existing_report, trained_existing = train_family(existing_rows, "next", base_existing_features)

    demo_path = Path("public/data/demo-data.json")
    demo = json.loads(demo_path.read_text()) if demo_path.exists() else {"products": []}
    demo_predictions = demo_existing_predictions(demo.get("products", []), trained_existing, price_map, store_map, trend_rows)
    new_predictions = new_product_predictions(demo.get("newProductModel", {}).get("attributeOptions", {}), trained_new)

    report = {
        "generatedAt": datetime.utcnow().isoformat() + "Z",
        "modelType": "XGBoost Regressor",
        "newProductModel": {
            "problemDefinition": {
                "userInput": ["category", "color", "fabric"],
                "outputs": ["future4wDemand", "future8wDemand", "future12wDemand"],
            },
            "samples": len(new_rows),
            **new_report,
        },
        "existingProductModel": {
            "problemDefinition": {
                "userInput": "product_id",
                "lookupBehavior": "product_id retrieves attributes, actual sales momentum, price/discount, store/restock signals, category benchmarks, and Google Trends",
                "outputs": ["next4wDemand", "next8wDemand", "next12wDemand"],
            },
            "samples": len(existing_rows),
            **existing_report,
        },
    }

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(json.dumps(report, indent=2) + "\n")

    artifact = {
        "generatedAt": report["generatedAt"],
        "modelType": "XGBoost Regressor",
        "userInput": "product_id",
        "outputs": ["next4wDemand", "next8wDemand", "next12wDemand"],
        "predictionsByProductId": demo_predictions,
        "validationSummary": {
            horizon: {
                "testWapePct": existing_report["horizons"][horizon]["split"]["test"]["wapePct"],
                "testAccuracyPct": existing_report["horizons"][horizon]["split"]["test"]["accuracyPct"],
                "testBiasPct": existing_report["horizons"][horizon]["split"]["test"]["biasPct"],
            }
            for horizon in map(str, HORIZONS)
        },
    }
    OUT_EXISTING_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    OUT_EXISTING_ARTIFACT.write_text(json.dumps(artifact, indent=2) + "\n")

    new_artifact = {
        "generatedAt": report["generatedAt"],
        "modelType": "XGBoost Regressor",
        "userInput": ["category", "color", "fabric"],
        "outputs": ["future4wDemand", "future8wDemand", "future12wDemand"],
        "predictionsByKey": new_predictions,
        "categoryLaunchBenchmarks": category_launch_benchmarks(new_rows),
        "validationSummary": {
            horizon: {
                "testWapePct": new_report["horizons"][horizon]["split"]["test"]["wapePct"],
                "testAccuracyPct": new_report["horizons"][horizon]["split"]["test"]["accuracyPct"],
                "testBiasPct": new_report["horizons"][horizon]["split"]["test"]["biasPct"],
            }
            for horizon in map(str, HORIZONS)
        },
    }
    OUT_NEW_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    OUT_NEW_ARTIFACT.write_text(json.dumps(new_artifact, indent=2) + "\n")

    print(f"Wrote {OUT_REPORT}")
    print(f"Wrote {OUT_EXISTING_ARTIFACT}")
    print(f"Wrote {OUT_NEW_ARTIFACT}")
    for family, family_report in [("New", new_report), ("Existing", existing_report)]:
        for horizon in map(str, HORIZONS):
            test = family_report["horizons"][horizon]["split"]["test"]
            print(f"{family} {horizon}w test WAPE: {test['wapePct']}%")
            print(f"{family} {horizon}w test accuracy: {test['accuracyPct']}%")


if __name__ == "__main__":
    main()
