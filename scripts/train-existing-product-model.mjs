import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

const MYSQL = process.env.MYSQL_BIN || "/Applications/MAMP/Library/bin/mysql80/bin/mysql";
const DB_HOST = process.env.DB_HOST || "127.0.0.1";
const DB_USER = process.env.DB_USER || "root";
const DB_PASSWORD = process.env.DB_PASSWORD || "root";
const DB_NAME = process.env.DB_NAME || "visuelle";
const OUTPUT = process.env.EXISTING_MODEL_REPORT || "public/data/existing-model-validation.json";
const MODEL_OUTPUT = process.env.EXISTING_MODEL_ARTIFACT || "public/data/existing-demand-model.json";
const SEED = Number(process.env.MODEL_SEED || 580);
const OBSERVED_WEEKS = 4;
const HORIZONS = [4, 8, 12];

const query = `
SELECT
  external_code,
  COALESCE(category, '') AS category,
  COALESCE(color, '') AS color,
  COALESCE(fabric, '') AS fabric,
  COALESCE(release_date, '1970-01-01') AS release_date,
  week_since_release,
  units
FROM product_weekly_lifecycle
WHERE external_code IS NOT NULL
  AND category IS NOT NULL
  AND color IS NOT NULL
  AND fabric IS NOT NULL
  AND week_since_release IS NOT NULL
  AND week_since_release BETWEEN 0 AND 15
  AND units IS NOT NULL
ORDER BY external_code, week_since_release;
`;

const priceQuery = `
SELECT
  external_code,
  AVG(price) AS avg_price,
  AVG((COALESCE(week_0, 0) + COALESCE(week_1, 0) + COALESCE(week_2, 0) + COALESCE(week_3, 0)) / 4) AS avg_discount_w0_w3,
  MAX(GREATEST(COALESCE(week_0, 0), COALESCE(week_1, 0), COALESCE(week_2, 0), COALESCE(week_3, 0))) AS max_discount_w0_w3
FROM price_discount_series
WHERE external_code IS NOT NULL
GROUP BY external_code;
`;

const storeRestockQuery = `
SELECT
  external_code,
  COUNT(DISTINCT retail) AS store_coverage,
  SUM(COALESCE(restock, 0)) AS restock_signal,
  SUM(CASE WHEN COALESCE(restock, 0) > 0 THEN 1 ELSE 0 END) AS restock_store_rows
FROM sales
WHERE external_code IS NOT NULL
GROUP BY external_code;
`;

const gtrendQuery = "SELECT * FROM vis2_gtrends_data ORDER BY date;";

function mysqlTsv(sql) {
  return execFileSync(MYSQL, [
    `-h${DB_HOST}`,
    `-u${DB_USER}`,
    `-p${DB_PASSWORD}`,
    "--batch",
    "--raw",
    "--skip-column-names",
    DB_NAME,
    "-e",
    sql,
  ], { encoding: "utf8", maxBuffer: 1024 * 1024 * 64 });
}

function mysqlTsvWithHeader(sql) {
  return execFileSync(MYSQL, [
    `-h${DB_HOST}`,
    `-u${DB_USER}`,
    `-p${DB_PASSWORD}`,
    "--batch",
    "--raw",
    DB_NAME,
    "-e",
    sql,
  ], { encoding: "utf8", maxBuffer: 1024 * 1024 * 64 });
}

function parseRows(tsv) {
  return tsv.trim().split("\n").filter(Boolean).map((line) => {
    const [externalCode, category, color, fabric, releaseDate, week, units] = line.split("\t");
    return { externalCode, category, color, fabric, releaseDate, week: Number(week), units: Number(units) };
  }).filter((row) => Number.isFinite(row.week) && Number.isFinite(row.units));
}

function parseNumericMap(tsv, fields) {
  const map = new Map();
  tsv.trim().split("\n").filter(Boolean).forEach((line) => {
    const parts = line.split("\t");
    const id = parts[0];
    const values = {};
    fields.forEach((field, index) => {
      const value = Number(parts[index + 1]);
      values[field] = Number.isFinite(value) ? value : 0;
    });
    map.set(id, values);
  });
  return map;
}

function slug(value) {
  return String(value || "").trim().toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
}

function parseGtrends(tsv) {
  const lines = tsv.trim().split("\n").filter(Boolean);
  const headers = lines.shift().split("\t");
  return lines.map((line) => {
    const parts = line.split("\t");
    const row = {};
    headers.forEach((header, index) => {
      row[header] = index === 0 ? parts[index] : Number(parts[index]) || 0;
    });
    return row;
  });
}

function nearestTrendRow(gtrendRows, releaseDate) {
  const date = new Date(releaseDate);
  if (Number.isNaN(date.getTime()) || !gtrendRows.length) return null;
  let best = gtrendRows[0];
  let bestDistance = Infinity;
  gtrendRows.forEach((row) => {
    const distance = Math.abs(new Date(row.date).getTime() - date.getTime());
    if (distance < bestDistance) {
      best = row;
      bestDistance = distance;
    }
  });
  return best;
}

function trendValue(row, key) {
  if (!row || !key) return 0;
  return Number(row[slug(key)] || 0);
}

function productSamples(rows, priceMap, storeRestockMap, gtrendRows) {
  const byProduct = new Map();
  rows.forEach((row) => {
    if (!byProduct.has(row.externalCode)) {
      byProduct.set(row.externalCode, {
        externalCode: row.externalCode,
        category: row.category,
        color: row.color,
        fabric: row.fabric,
        releaseDate: row.releaseDate,
        weekly: Array(16).fill(null),
      });
    }
    byProduct.get(row.externalCode).weekly[row.week] = row.units;
  });

  return [...byProduct.values()]
    .filter((sample) => sample.weekly.slice(0, 16).every((value) => Number.isFinite(value)))
    .map((sample) => {
      const observed = sample.weekly.slice(0, OBSERVED_WEEKS);
      sample.observed4wUnits = observed.reduce((sum, value) => sum + value, 0);
      sample.observedFirst2wUnits = observed.slice(0, 2).reduce((sum, value) => sum + value, 0);
      sample.observedLast2wUnits = observed.slice(2, 4).reduce((sum, value) => sum + value, 0);
      sample.observedTrendRatio = sample.observedFirst2wUnits > 0
        ? sample.observedLast2wUnits / sample.observedFirst2wUnits
        : 1;
      sample.week0Units = sample.weekly[0];
      sample.week1Units = sample.weekly[1];
      sample.week2Units = sample.weekly[2];
      sample.week3Units = sample.weekly[3];
      const price = priceMap.get(sample.externalCode) || {};
      const storeRestock = storeRestockMap.get(sample.externalCode) || {};
      const trendRow = nearestTrendRow(gtrendRows, sample.releaseDate);
      sample.avgPrice = price.avgPrice || 0;
      sample.avgDiscountW0W3 = price.avgDiscountW0W3 || 0;
      sample.maxDiscountW0W3 = price.maxDiscountW0W3 || 0;
      sample.storeCoverage = storeRestock.storeCoverage || 0;
      sample.restockSignal = storeRestock.restockSignal || 0;
      sample.restockStoreRows = storeRestock.restockStoreRows || 0;
      sample.categoryTrend = trendValue(trendRow, sample.category);
      sample.colorTrend = trendValue(trendRow, sample.color);
      sample.fabricTrend = trendValue(trendRow, sample.fabric);
      sample.combinedTrend = (sample.categoryTrend * 0.5) + (sample.colorTrend * 0.25) + (sample.fabricTrend * 0.25);
      HORIZONS.forEach((horizon) => {
        sample[`next${horizon}wDemand`] = sample.weekly
          .slice(OBSERVED_WEEKS, OBSERVED_WEEKS + horizon)
          .reduce((sum, value) => sum + value, 0);
      });
      return sample;
    });
}

function hashString(value, seed = SEED) {
  let hash = 2166136261 ^ seed;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function splitSamples(samples) {
  const ordered = [...samples].map((sample) => ({ ...sample, hash: hashString(String(sample.externalCode)) }))
    .sort((a, b) => a.hash - b.hash);
  const trainEnd = Math.floor(ordered.length * 0.7);
  const validationEnd = trainEnd + Math.floor(ordered.length * 0.15);
  return ordered.map((sample, index) => ({
    ...sample,
    split: index < trainEnd ? "train" : index < validationEnd ? "validation" : "test",
  }));
}

function topValues(samples, field, limit = 100) {
  const counts = new Map();
  samples.forEach((sample) => counts.set(sample[field], (counts.get(sample[field]) || 0) + 1));
  return [...counts.entries()]
    .filter(([value]) => value !== "")
    .sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0])))
    .slice(0, limit)
    .map(([value]) => value);
}

function buildSpec(trainSamples) {
  const fields = ["category", "color", "fabric"];
  const valueSets = {};
  const featureNames = [
    "intercept",
    "log_observed4w_units",
    "log_observed_first2w_units",
    "log_observed_last2w_units",
    "observed_trend_ratio_capped",
    "week0_share",
    "week1_share",
    "week2_share",
    "week3_share",
    "avg_price",
    "avg_discount_w0_w3",
    "max_discount_w0_w3",
    "log_store_coverage",
    "log_restock_signal",
    "log_restock_store_rows",
    "category_trend",
    "color_trend",
    "fabric_trend",
    "combined_trend",
  ];
  fields.forEach((field) => {
    valueSets[field] = new Set(topValues(trainSamples, field, field === "fabric" ? 120 : 100));
    valueSets[field].forEach((value) => featureNames.push(`${field}=${value}`));
    featureNames.push(`${field}=__OTHER__`);
  });
  return { fields, valueSets, featureNames };
}

function buildHorizonSpec(baseSpec, trainSamples, targetField) {
  const categoryStats = new Map();
  const globalMean = trainSamples.reduce((sum, sample) => sum + sample[targetField], 0) / Math.max(1, trainSamples.length);

  trainSamples.forEach((sample) => {
    const current = categoryStats.get(sample.category) || { sum: 0, count: 0 };
    current.sum += sample[targetField];
    current.count += 1;
    categoryStats.set(sample.category, current);
  });

  return {
    ...baseSpec,
    featureNames: [
      ...baseSpec.featureNames,
      "category_benchmark_ratio",
      "category_benchmark_count_log",
    ],
    categoryBenchmark: {
      globalMean,
      priorWeight: 20,
      stats: categoryStats,
    },
  };
}

function categoryBenchmarkVector(sample, spec) {
  if (!spec.categoryBenchmark) return [];
  const stat = spec.categoryBenchmark.stats.get(sample.category) || { sum: 0, count: 0 };
  const benchmark = (
    stat.sum + spec.categoryBenchmark.globalMean * spec.categoryBenchmark.priorWeight
  ) / (stat.count + spec.categoryBenchmark.priorWeight);
  return [
    benchmark / Math.max(1, spec.categoryBenchmark.globalMean),
    Math.log1p(stat.count),
  ];
}

function vectorize(sample, spec) {
  const total = Math.max(1, sample.observed4wUnits);
  const vector = [
    1,
    Math.log1p(sample.observed4wUnits),
    Math.log1p(sample.observedFirst2wUnits),
    Math.log1p(sample.observedLast2wUnits),
    Math.min(5, Math.max(0, sample.observedTrendRatio)),
    sample.week0Units / total,
    sample.week1Units / total,
    sample.week2Units / total,
    sample.week3Units / total,
    sample.avgPrice,
    sample.avgDiscountW0W3,
    sample.maxDiscountW0W3,
    Math.log1p(sample.storeCoverage),
    Math.log1p(sample.restockSignal),
    Math.log1p(sample.restockStoreRows),
    sample.categoryTrend / 100,
    sample.colorTrend / 100,
    sample.fabricTrend / 100,
    sample.combinedTrend / 100,
  ];
  spec.fields.forEach((field) => {
    const values = spec.valueSets[field];
    values.forEach((value) => vector.push(sample[field] === value ? 1 : 0));
    vector.push(values.has(sample[field]) ? 0 : 1);
  });
  vector.push(...categoryBenchmarkVector(sample, spec));
  return vector;
}

function addOuterProduct(xtx, xty, x, y) {
  for (let i = 0; i < x.length; i += 1) {
    xty[i] += x[i] * y;
    for (let j = 0; j <= i; j += 1) xtx[i][j] += x[i] * x[j];
  }
}

function solveLinearSystem(matrix, vector) {
  const n = vector.length;
  const a = matrix.map((row, i) => [...row, vector[i]]);
  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < n; row += 1) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    }
    if (Math.abs(a[pivot][col]) < 1e-12) continue;
    if (pivot !== col) [a[col], a[pivot]] = [a[pivot], a[col]];
    const divisor = a[col][col];
    for (let j = col; j <= n; j += 1) a[col][j] /= divisor;
    for (let row = 0; row < n; row += 1) {
      if (row === col) continue;
      const factor = a[row][col];
      if (!factor) continue;
      for (let j = col; j <= n; j += 1) a[row][j] -= factor * a[col][j];
    }
  }
  return a.map((row) => row[n] || 0);
}

function trainRidge(samples, spec, targetField, lambda, targetMode) {
  const p = spec.featureNames.length;
  const xtx = Array.from({ length: p }, () => Array(p).fill(0));
  const xty = Array(p).fill(0);
  samples.forEach((sample) => {
    const x = vectorize(sample, spec);
    const target = sample[targetField];
    const y = targetMode === "log1p" ? Math.log1p(target) : targetMode === "sqrt" ? Math.sqrt(target) : target;
    addOuterProduct(xtx, xty, x, y);
  });
  for (let i = 0; i < p; i += 1) {
    for (let j = i + 1; j < p; j += 1) xtx[i][j] = xtx[j][i];
    if (i > 0) xtx[i][i] += lambda;
  }
  return solveLinearSystem(xtx, xty);
}

function predict(sample, spec, weights, targetMode, calibration = 1) {
  const x = vectorize(sample, spec);
  const raw = x.reduce((sum, value, index) => sum + value * weights[index], 0);
  const prediction = targetMode === "log1p" ? Math.expm1(raw) : targetMode === "sqrt" ? raw ** 2 : raw;
  return Math.max(0, prediction * calibration);
}

function calibrationFactor(samples, spec, weights, targetField, targetMode) {
  const actualTotal = samples.reduce((sum, sample) => sum + sample[targetField], 0);
  const forecastTotal = samples.reduce((sum, sample) => sum + predict(sample, spec, weights, targetMode), 0);
  return forecastTotal > 0 ? actualTotal / forecastTotal : 1;
}

function metrics(samples, spec, weights, targetField, targetMode, calibration = 1) {
  let absoluteError = 0;
  let squaredError = 0;
  let signedError = 0;
  let actualTotal = 0;
  let forecastTotal = 0;
  samples.forEach((sample) => {
    const forecast = predict(sample, spec, weights, targetMode, calibration);
    const actual = sample[targetField];
    const error = forecast - actual;
    absoluteError += Math.abs(error);
    squaredError += error ** 2;
    signedError += error;
    actualTotal += actual;
    forecastTotal += forecast;
  });
  const wape = absoluteError / Math.max(1, actualTotal);
  return {
    products: samples.length,
    actualUnits: Math.round(actualTotal),
    forecastUnits: Math.round(forecastTotal),
    mae: Number((absoluteError / Math.max(1, samples.length)).toFixed(2)),
    rmse: Number(Math.sqrt(squaredError / Math.max(1, samples.length)).toFixed(2)),
    wape: Number(wape.toFixed(4)),
    wapePct: Number((wape * 100).toFixed(2)),
    accuracyPct: Number(Math.max(0, (1 - wape) * 100).toFixed(2)),
    bias: Number((signedError / Math.max(1, actualTotal)).toFixed(4)),
    biasPct: Number(((signedError / Math.max(1, actualTotal)) * 100).toFixed(2)),
  };
}

function trainHorizonModel({ horizon, trainSamples, validationSamples, testSamples, spec }) {
  const targetField = `next${horizon}wDemand`;
  const horizonSpec = buildHorizonSpec(spec, trainSamples, targetField);
  const lambdas = [0.1, 1, 5, 10, 25, 50, 100, 250, 500];
  const targetModes = ["log1p", "sqrt", "raw"];
  const candidates = targetModes.flatMap((targetMode) => lambdas.flatMap((lambda) => {
    const weights = trainRidge(trainSamples, horizonSpec, targetField, lambda, targetMode);
    const validationCalibration = calibrationFactor(validationSamples, horizonSpec, weights, targetField, targetMode);
    const rawValidation = metrics(validationSamples, horizonSpec, weights, targetField, targetMode);
    return [
      { targetMode, lambda, calibrated: false, calibration: 1, weights, validationRaw: rawValidation, validation: rawValidation },
      {
        targetMode,
        lambda,
        calibrated: true,
        calibration: validationCalibration,
        weights,
        validationRaw: rawValidation,
        validation: metrics(validationSamples, horizonSpec, weights, targetField, targetMode, validationCalibration),
      },
    ];
  }));
  const best = candidates.reduce((winner, candidate) => (
    candidate.validation.wape < winner.validation.wape ? candidate : winner
  ), candidates[0]);
  return {
    horizonWeeks: horizon,
    target: `next ${horizon}-week demand from W${OBSERVED_WEEKS}-W${OBSERVED_WEEKS + horizon - 1}`,
    selectedModel: {
      type: "Ridge regression",
      targetMode: best.targetMode,
      lambda: best.lambda,
      calibratedOnValidation: best.calibrated,
      validationCalibrationFactor: Number(best.calibration.toFixed(4)),
    },
    artifact: {
      horizonWeeks: horizon,
      targetMode: best.targetMode,
      calibration: best.calibration,
      weights: best.weights,
      featureNames: horizonSpec.featureNames,
      categoryBenchmark: {
        globalMean: horizonSpec.categoryBenchmark.globalMean,
        priorWeight: horizonSpec.categoryBenchmark.priorWeight,
        stats: Object.fromEntries([...horizonSpec.categoryBenchmark.stats.entries()]),
      },
    },
    split: {
      train: metrics(trainSamples, horizonSpec, best.weights, targetField, best.targetMode, best.calibration),
      validation: best.validation,
      test: metrics(testSamples, horizonSpec, best.weights, targetField, best.targetMode, best.calibration),
    },
    validationTuning: candidates.map((candidate) => ({
      targetMode: candidate.targetMode,
      lambda: candidate.lambda,
      calibrated: candidate.calibrated,
      calibrationFactor: Number(candidate.calibration.toFixed(4)),
      validationRawWapePct: candidate.validationRaw.wapePct,
      validationRawBiasPct: candidate.validationRaw.biasPct,
      validationWapePct: candidate.validation.wapePct,
      validationAccuracyPct: candidate.validation.accuracyPct,
      validationBiasPct: candidate.validation.biasPct,
      validationMae: candidate.validation.mae,
      validationRmse: candidate.validation.rmse,
    })).sort((a, b) => a.validationWapePct - b.validationWapePct),
  };
}

const rows = parseRows(mysqlTsv(query));
const priceMap = parseNumericMap(mysqlTsv(priceQuery), ["avgPrice", "avgDiscountW0W3", "maxDiscountW0W3"]);
const storeRestockMap = parseNumericMap(mysqlTsv(storeRestockQuery), ["storeCoverage", "restockSignal", "restockStoreRows"]);
const gtrendRows = parseGtrends(mysqlTsvWithHeader(gtrendQuery));
const samples = splitSamples(productSamples(rows, priceMap, storeRestockMap, gtrendRows));
const trainSamples = samples.filter((sample) => sample.split === "train");
const validationSamples = samples.filter((sample) => sample.split === "validation");
const testSamples = samples.filter((sample) => sample.split === "test");
const spec = buildSpec(trainSamples);
const horizons = HORIZONS.map((horizon) => trainHorizonModel({
  horizon,
  trainSamples,
  validationSamples,
  testSamples,
  spec,
}));

const report = {
  generatedAt: new Date().toISOString(),
  source: {
    database: DB_NAME,
    primaryTable: "product_weekly_lifecycle",
    auxiliaryTables: ["price_discount_series", "sales", "vis2_gtrends_data"],
    rawRowsUsed: rows.length,
    productSamples: samples.length,
  },
  problemDefinition: {
    observedWindow: "W0-W3 actual sales are visible to the model",
    userInput: "product_id",
    lookupBehavior: "The system uses product_id to retrieve product attributes, actual sales momentum, price/discount, store/restock signals, category benchmark, and Google Trends.",
    internalFeatures: [
      "category",
      "color",
      "fabric",
      "observed W0-W3 total units",
      "observed first-2-week units",
      "observed last-2-week units",
      "observed sales trend ratio",
      "weekly sales mix across W0-W3",
      "average price",
      "observed W0-W3 average discount",
      "observed W0-W3 maximum discount",
      "store coverage",
      "restock signal",
      "category benchmark",
      "Google Trends category/color/fabric release-period signals",
    ],
    outputs: [
      "next 4-week total demand",
      "next 8-week total demand",
      "next 12-week total demand",
    ],
  },
  splitPolicy: {
    method: "Product-level deterministic 70/15/15 split by external_code hash",
    trainProducts: trainSamples.length,
    validationProducts: validationSamples.length,
    testProducts: testSamples.length,
  },
  model: {
    type: "Ridge regression, one model per forecast horizon",
    candidateTargetTransforms: ["raw units", "sqrt(units), squared back for metrics", "log1p(units), converted back with expm1"],
    features: spec.featureNames,
  },
  horizons,
};

mkdirSync(dirname(resolve(OUTPUT)), { recursive: true });
writeFileSync(OUTPUT, `${JSON.stringify(report, null, 2)}\n`);

const artifact = {
  generatedAt: report.generatedAt,
  userInput: "product_id",
  outputs: ["next4wDemand", "next8wDemand", "next12wDemand"],
  observedWeeks: OBSERVED_WEEKS,
  internalFeatures: report.problemDefinition.internalFeatures,
  baseSpec: {
    fields: spec.fields,
    valueSets: Object.fromEntries(Object.entries(spec.valueSets).map(([field, values]) => [field, [...values]])),
    featureNames: spec.featureNames,
  },
  horizons: Object.fromEntries(horizons.map((model) => [String(model.horizonWeeks), model.artifact])),
  validationSummary: Object.fromEntries(horizons.map((model) => [String(model.horizonWeeks), {
    testWapePct: model.split.test.wapePct,
    testAccuracyPct: model.split.test.accuracyPct,
    testBiasPct: model.split.test.biasPct,
  }])),
};

mkdirSync(dirname(resolve(MODEL_OUTPUT)), { recursive: true });
writeFileSync(MODEL_OUTPUT, `${JSON.stringify(artifact, null, 2)}\n`);

console.log(`Wrote ${OUTPUT}`);
console.log(`Wrote ${MODEL_OUTPUT}`);
horizons.forEach((model) => {
  console.log(`${model.horizonWeeks}w test WAPE: ${model.split.test.wapePct}%`);
  console.log(`${model.horizonWeeks}w test accuracy: ${model.split.test.accuracyPct}%`);
  console.log(`${model.horizonWeeks}w test bias: ${model.split.test.biasPct}%`);
});
