#!/usr/bin/env python3
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "train-xgboost-models.py"
IMAGE_DIR = ROOT / "public" / "product-images"
OUT_JSON = ROOT / "experiments" / "new-product-image-embedding-xgboost-results.json"


def load_base_module():
    spec = importlib.util.spec_from_file_location("visuelle_xgb", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def attribute_tokens(row):
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
        x_train = self.vectorizer.fit_transform([attribute_tokens(row) for row in train_rows])
        self.dims = max(1, min(requested_dims, x_train.shape[0] - 1, x_train.shape[1] - 1))
        self.model = TruncatedSVD(n_components=self.dims, random_state=580)
        self.model.fit(x_train)

    def features(self, row):
        values = self.model.transform(self.vectorizer.transform([attribute_tokens(row)]))[0]
        return {f"attr_embed_{idx}": float(value) for idx, value in enumerate(values)}


def image_path(product_id):
    return IMAGE_DIR / f"{product_id}.png"


def raw_image_vector(product_id):
    path = image_path(product_id)
    if not path.exists():
        return None

    with Image.open(path) as img:
        rgba = img.convert("RGBA")
        rgb = rgba.convert("RGB").resize((32, 32))
        pixels = np.asarray(rgb, dtype=np.float32) / 255.0
        alpha = np.asarray(rgba.getchannel("A"), dtype=np.float32) / 255.0
        full_rgb = np.asarray(rgba.convert("RGB"), dtype=np.float32) / 255.0

    flat_pixels = pixels.reshape(-1)
    hist_parts = []
    for channel in range(3):
        hist, _ = np.histogram(pixels[:, :, channel], bins=12, range=(0, 1), density=True)
        hist_parts.append(hist.astype(np.float32))

    whiteness = full_rgb.mean(axis=2)
    foreground_mask = (alpha > 0.15) & (whiteness < 0.95)
    foreground_share = np.array([foreground_mask.mean()], dtype=np.float32)
    if foreground_mask.any():
        ys, xs = np.where(foreground_mask)
        bbox_width = (xs.max() - xs.min() + 1) / max(1, foreground_mask.shape[1])
        bbox_height = (ys.max() - ys.min() + 1) / max(1, foreground_mask.shape[0])
        aspect = bbox_height / max(0.01, bbox_width)
        mean_rgb = full_rgb[foreground_mask].mean(axis=0)
    else:
        bbox_width = bbox_height = aspect = 0.0
        mean_rgb = np.zeros(3, dtype=np.float32)

    shape_features = np.array([bbox_width, bbox_height, aspect], dtype=np.float32)
    return np.concatenate([flat_pixels, *hist_parts, foreground_share, shape_features, mean_rgb])


class ImageSvdEmbedding:
    def __init__(self, train_rows, requested_dims):
        self.default_vector = None
        train_vectors = []
        for row in train_rows:
            vector = raw_image_vector(row["external_code"])
            if vector is not None:
                train_vectors.append(vector)
        if not train_vectors:
            raise RuntimeError("No product images found for image embedding experiment.")

        raw = np.vstack(train_vectors)
        self.default_vector = raw.mean(axis=0)
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(raw)
        self.dims = max(1, min(requested_dims, scaled.shape[0] - 1, scaled.shape[1] - 1))
        self.model = TruncatedSVD(n_components=self.dims, random_state=580)
        self.model.fit(scaled)
        self.cache = {}

    def vector_for(self, product_id):
        if product_id not in self.cache:
            self.cache[product_id] = raw_image_vector(product_id)
        return self.cache[product_id] if self.cache[product_id] is not None else self.default_vector

    def features(self, row):
        raw = self.vector_for(row["external_code"]).reshape(1, -1)
        values = self.model.transform(self.scaler.transform(raw))[0]
        return {f"image_embed_{idx}": float(value) for idx, value in enumerate(values)}


def train_variant(base, rows, variant):
    split_map = base.split_ids(sorted({row["external_code"] for row in rows}))
    working_rows = [{**row, "split": split_map[row["external_code"]]} for row in rows]
    train_rows = [row for row in working_rows if row["split"] == "train"]
    val_rows = [row for row in working_rows if row["split"] == "validation"]
    test_rows = [row for row in working_rows if row["split"] == "test"]
    attr_embedder = AttributeSvdEmbedding(train_rows, variant["attrDims"]) if variant["attrDims"] else None
    image_embedder = ImageSvdEmbedding(train_rows, variant["imageDims"]) if variant["imageDims"] else None

    horizons = {}
    for horizon in base.HORIZONS:
        target = f"future{horizon}wDemand"
        group_model = base.build_group_means(train_rows, target)

        def feature_fn(row, gm=group_model, attr=attr_embedder, image=image_embedder):
            features = base.base_new_features(row, gm)
            if attr:
                features.update(attr.features(row))
            if image:
                features.update(image.features(row))
            return features

        trained = base.train_xgb(train_rows, val_rows, test_rows, feature_fn, target)
        horizons[str(horizon)] = {
            "target": target,
            "selectedParams": trained["params"],
            "attrDimsUsed": attr_embedder.dims if attr_embedder else 0,
            "imageDimsUsed": image_embedder.dims if image_embedder else 0,
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
        "imageCoverage": {
            "allRowsWithImage": sum(1 for row in working_rows if image_path(row["external_code"]).exists()),
            "totalRows": len(working_rows),
        },
        "horizons": horizons,
    }


def average_metric(result, split, metric):
    values = [horizon["split"][split][metric] for horizon in result["horizons"].values()]
    return round(sum(values) / max(1, len(values)), 2)


def summarize(result):
    return {
        "name": result["variant"]["name"],
        "attrDims": result["variant"]["attrDims"],
        "imageDims": result["variant"]["imageDims"],
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
        {"name": "structured_xgboost_baseline", "attrDims": 0, "imageDims": 0},
        {"name": "structured_plus_image_svd_8", "attrDims": 0, "imageDims": 8},
        {"name": "structured_plus_image_svd_16", "attrDims": 0, "imageDims": 16},
        {"name": "structured_plus_image_svd_32", "attrDims": 0, "imageDims": 32},
        {"name": "structured_plus_attr4_image16", "attrDims": 4, "imageDims": 16},
    ]
    results = [train_variant(base, rows, variant) for variant in variants]
    summary = [summarize(result) for result in results]
    baseline = next(item for item in summary if item["name"] == "structured_xgboost_baseline")
    best_by_validation = min(summary, key=lambda item: item["avgValidationWapePct"])

    report = {
        "generatedAt": datetime.utcnow().isoformat() + "Z",
        "experiment": "New product XGBoost with image embeddings plus structured features",
        "doesNotModifyAppModel": True,
        "targetDefinition": {
            "input": ["category", "color", "fabric", "product image"],
            "outputs": ["future4wDemand", "future8wDemand", "future12wDemand"],
            "meaning": "Cumulative launch demand from W0 through each horizon.",
        },
        "imageEmbeddingDefinition": {
            "type": "TruncatedSVD over image descriptors",
            "rawImageDescriptor": "32x32 RGB pixels, RGB histograms, foreground share, foreground bounding box, aspect ratio, and foreground mean color",
            "training": "Scaler and SVD are fit on training product images only inside the same 70/15/15 split.",
            "note": "This is a local image embedding experiment. It does not download or fine-tune a pretrained vision model.",
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
