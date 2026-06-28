#!/usr/bin/env python3
import importlib.util
import json
import math
from datetime import datetime
from pathlib import Path

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "train-xgboost-models.py"
OUT_JSON = ROOT / "experiments" / "new-product-embedding-xgboost-results.json"


def load_base_module():
    spec = importlib.util.spec_from_file_location("visuelle_xgb", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def embedding_tokens(row):
    category = row["category"]
    color = row["color"]
    fabric = row["fabric"]
    return {
        f"category={category}": 1,
        f"color={color}": 1,
        f"fabric={fabric}": 1,
        f"category_color={category}||{color}": 1,
        f"category_fabric={category}||{fabric}": 1,
        f"color_fabric={color}||{fabric}": 1,
        f"category_color_fabric={category}||{color}||{fabric}": 1,
    }


class AttributeSvdEmbedding:
    def __init__(self, train_rows, requested_dims):
        self.vectorizer = DictVectorizer(sparse=True)
        x_train = self.vectorizer.fit_transform([embedding_tokens(row) for row in train_rows])
        max_dims = max(1, min(requested_dims, x_train.shape[0] - 1, x_train.shape[1] - 1))
        self.dims = max_dims
        self.model = TruncatedSVD(n_components=max_dims, random_state=580)
        self.model.fit(x_train)

    def features(self, row):
        x = self.vectorizer.transform([embedding_tokens(row)])
        values = self.model.transform(x)[0]
        return {f"attr_embed_{idx}": float(value) for idx, value in enumerate(values)}


def train_variant(base, rows, variant):
    split_map = base.split_ids(sorted({row["external_code"] for row in rows}))
    working_rows = [{**row, "split": split_map[row["external_code"]]} for row in rows]
    train_rows = [row for row in working_rows if row["split"] == "train"]
    val_rows = [row for row in working_rows if row["split"] == "validation"]
    test_rows = [row for row in working_rows if row["split"] == "test"]
    embedder = AttributeSvdEmbedding(train_rows, variant["embeddingDims"]) if variant["embeddingDims"] else None

    horizons = {}
    for horizon in base.HORIZONS:
        target = f"future{horizon}wDemand"
        group_model = base.build_group_means(train_rows, target)

        def feature_fn(row, gm=group_model, emb=embedder):
            features = base.base_new_features(row, gm)
            if emb:
                features.update(emb.features(row))
            return features

        trained = base.train_xgb(train_rows, val_rows, test_rows, feature_fn, target)
        horizons[str(horizon)] = {
            "target": target,
            "selectedParams": trained["params"],
            "embeddingDimsUsed": embedder.dims if embedder else 0,
            "split": trained["split"],
        }

    return {
        "variant": variant,
        "splitPolicy": {
            "method": "Product-level deterministic 70/15/15 split by external_code hash",
            "trainProducts": len({row["external_code"] for row in train_rows}),
            "validationProducts": len({row["external_code"] for row in val_rows}),
            "testProducts": len({row["external_code"] for row in test_rows}),
            "trainSamples": len(train_rows),
            "validationSamples": len(val_rows),
            "testSamples": len(test_rows),
        },
        "horizons": horizons,
    }


def average_metric(result, split, metric):
    values = [horizon["split"][split][metric] for horizon in result["horizons"].values()]
    return round(sum(values) / max(1, len(values)), 2)


def summarize(result):
    return {
        "name": result["variant"]["name"],
        "embeddingDims": result["variant"]["embeddingDims"],
        "avgValidationWapePct": average_metric(result, "validation", "wapePct"),
        "avgTestWapePct": average_metric(result, "test", "wapePct"),
        "avgTestAccuracyPct": average_metric(result, "test", "accuracyPct"),
        "horizonTest": {
            horizon: {
                "wapePct": payload["split"]["test"]["wapePct"],
                "accuracyPct": payload["split"]["test"]["accuracyPct"],
                "biasPct": payload["split"]["test"]["biasPct"],
            }
            for horizon, payload in result["horizons"].items()
        },
    }


def main():
    base = load_base_module()
    products = base.load_base_rows()
    rows = base.make_new_product_rows(products)
    variants = [
        {"name": "structured_xgboost_baseline", "embeddingDims": 0},
        {"name": "structured_plus_attr_svd_4", "embeddingDims": 4},
        {"name": "structured_plus_attr_svd_8", "embeddingDims": 8},
        {"name": "structured_plus_attr_svd_16", "embeddingDims": 16},
    ]
    results = [train_variant(base, rows, variant) for variant in variants]
    summary = [summarize(result) for result in results]
    best_by_validation = min(summary, key=lambda item: item["avgValidationWapePct"])
    baseline = next(item for item in summary if item["name"] == "structured_xgboost_baseline")

    report = {
        "generatedAt": datetime.utcnow().isoformat() + "Z",
        "experiment": "New product XGBoost with attribute SVD embeddings plus structured features",
        "doesNotModifyAppModel": True,
        "targetDefinition": {
            "input": ["category", "color", "fabric"],
            "outputs": ["future4wDemand", "future8wDemand", "future12wDemand"],
            "meaning": "Cumulative launch demand from W0 through each horizon.",
        },
        "embeddingDefinition": {
            "type": "TruncatedSVD over attribute co-occurrence tokens",
            "tokens": ["category", "color", "fabric", "category_color", "category_fabric", "color_fabric", "category_color_fabric"],
            "training": "Unsupervised embedding fit on training products only inside the same 70/15/15 split.",
        },
        "baselineSummary": baseline,
        "bestByValidation": best_by_validation,
        "summary": summary,
        "details": results,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote {OUT_JSON}")
    print("Variant comparison:")
    for item in summary:
        delta = item["avgTestWapePct"] - baseline["avgTestWapePct"]
        sign = "+" if delta >= 0 else ""
        print(
            f"- {item['name']}: avg test WAPE {item['avgTestWapePct']}% "
            f"({sign}{round(delta, 2)} pts vs baseline), avg test accuracy {item['avgTestAccuracyPct']}%"
        )
    print(f"Best by validation: {best_by_validation['name']}")


if __name__ == "__main__":
    main()
