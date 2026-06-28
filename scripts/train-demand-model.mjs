import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

const MYSQL = process.env.MYSQL_BIN || "/Applications/MAMP/Library/bin/mysql80/bin/mysql";
const DB_HOST = process.env.DB_HOST || "127.0.0.1";
const DB_USER = process.env.DB_USER || "root";
const DB_PASSWORD = process.env.DB_PASSWORD || "root";
const DB_NAME = process.env.DB_NAME || "visuelle";
const OUTPUT = process.env.MODEL_REPORT || "public/data/model-validation.json";
const SEED = Number(process.env.MODEL_SEED || 580);
const HORIZONS = [4, 8, 12];

const query = `
SELECT
  l.external_code,
  COALESCE(l.category, '') AS category,
  COALESCE(l.color, '') AS color,
  COALESCE(l.fabric, '') AS fabric,
  COALESCE(DATE_FORMAT(pm.release_date, '%Y-%m-%d'), '') AS release_date,
  l.week_since_release,
  l.units
FROM product_weekly_lifecycle l
LEFT JOIN product_master pm ON pm.external_code = l.external_code
WHERE l.external_code IS NOT NULL
  AND l.category IS NOT NULL
  AND l.color IS NOT NULL
  AND l.fabric IS NOT NULL
  AND l.week_since_release IS NOT NULL
  AND l.week_since_release BETWEEN 0 AND 11
  AND l.units IS NOT NULL
ORDER BY l.external_code, l.week_since_release;
`;

const trendsQuery = "SELECT * FROM vis2_gtrends_data ORDER BY date;";

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

function parseRows(tsv) {
  return tsv.trim().split("\n").filter(Boolean).map((line) => {
    const [externalCode, category, color, fabric, releaseDate, week, units] = line.split("\t");
    return {
      externalCode,
      category,
      color,
      fabric,
      releaseDate,
      week: Number(week),
      units: Number(units),
    };
  }).filter((row) => Number.isFinite(row.week) && Number.isFinite(row.units));
}

function normalizeTrendKey(value) {
  return String(value || "").trim().toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
}

function parseTrendRows(tsv) {
  const lines = tsv.trim().split("\n").filter(Boolean);
  if (!lines.length) return [];
  const headers = lines[0].split("\t").map((header) => normalizeTrendKey(header));
  return lines.slice(1).map((line) => {
    const values = line.split("\t");
    const row = { date: values[0] };
    headers.slice(1).forEach((header, index) => {
      const value = Number(values[index + 1]);
      row[header] = Number.isFinite(value) && value >= 0 ? value : null;
    });
    return row;
  }).filter((row) => row.date);
}

function average(values) {
  const valid = values.filter((value) => Number.isFinite(value));
  return valid.length ? valid.reduce((sum, value) => sum + value, 0) / valid.length : null;
}

function trendFeaturesFor({ category, color, fabric, releaseDate }, trendRows) {
  const releaseTime = Date.parse(releaseDate);
  const preLaunchRows = Number.isFinite(releaseTime)
    ? trendRows.filter((row) => Date.parse(row.date) <= releaseTime)
    : trendRows;
  const recentRows = preLaunchRows.slice(-8);
  const priorRows = preLaunchRows.slice(-16, -8);
  const keys = {
    category: normalizeTrendKey(category),
    color: normalizeTrendKey(color),
    fabric: normalizeTrendKey(fabric),
  };
  const recent = {
    category: average(recentRows.map((row) => row[keys.category])),
    color: average(recentRows.map((row) => row[keys.color])),
    fabric: average(recentRows.map((row) => row[keys.fabric])),
  };
  const prior = {
    category: average(priorRows.map((row) => row[keys.category])),
    color: average(priorRows.map((row) => row[keys.color])),
    fabric: average(priorRows.map((row) => row[keys.fabric])),
  };
  const fallback = average(Object.values(recent)) ?? 50;
  const categoryTrend = recent.category ?? fallback;
  const colorTrend = recent.color ?? fallback;
  const fabricTrend = recent.fabric ?? fallback;
  const trendComposite = (categoryTrend * 0.5) + (colorTrend * 0.25) + (fabricTrend * 0.25);
  const priorComposite = (
    (prior.category ?? categoryTrend) * 0.5 +
    (prior.color ?? colorTrend) * 0.25 +
    (prior.fabric ?? fabricTrend) * 0.25
  );
  const trendMomentum = priorComposite > 0 ? trendComposite / priorComposite : 1;

  return {
    categoryTrend,
    colorTrend,
    fabricTrend,
    trendComposite,
    trendMomentum,
    trendCoverage: [recent.category, recent.color, recent.fabric].filter((value) => Number.isFinite(value)).length,
  };
}

function productSamples(rows, trendRows) {
  const byProduct = new Map();

  rows.forEach((row) => {
    if (!byProduct.has(row.externalCode)) {
      byProduct.set(row.externalCode, {
        externalCode: row.externalCode,
        category: row.category,
        color: row.color,
        fabric: row.fabric,
        releaseDate: row.releaseDate,
        weekly: Array(12).fill(null),
      });
    }
    byProduct.get(row.externalCode).weekly[row.week] = row.units;
  });

  return [...byProduct.values()]
    .filter((sample) => sample.weekly.slice(0, 12).every((value) => Number.isFinite(value)))
    .map((sample) => {
      HORIZONS.forEach((horizon) => {
        sample[`future${horizon}wDemand`] = sample.weekly
          .slice(0, horizon)
          .reduce((sum, value) => sum + value, 0);
      });
      return { ...sample, ...trendFeaturesFor(sample, trendRows) };
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
  const ordered = [...samples]
    .map((sample) => ({ ...sample, hash: hashString(String(sample.externalCode)) }))
    .sort((a, b) => a.hash - b.hash);
  const trainEnd = Math.floor(ordered.length * 0.7);
  const validationEnd = trainEnd + Math.floor(ordered.length * 0.15);

  return ordered.map((sample, index) => ({
    ...sample,
    split: index < trainEnd ? "train" : index < validationEnd ? "validation" : "test",
  }));
}

function topValues(samples, field, limit = 80) {
  const counts = new Map();
  samples.forEach((sample) => counts.set(sample[field], (counts.get(sample[field]) || 0) + 1));
  return [...counts.entries()]
    .filter(([value]) => value !== "")
    .sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0])))
    .slice(0, limit)
    .map(([value]) => value);
}

function buildVocabulary(trainSamples) {
  const fields = ["category", "color", "fabric"];
  const valueSets = {};
  const featureNames = [
    "intercept",
    "category_trend_recent_8w",
    "color_trend_recent_8w",
    "fabric_trend_recent_8w",
    "weighted_trend_recent_8w",
    "weighted_trend_momentum",
    "trend_coverage_count",
  ];

  fields.forEach((field) => {
    valueSets[field] = new Set(topValues(trainSamples, field, field === "fabric" ? 100 : 80));
    valueSets[field].forEach((value) => featureNames.push(`${field}=${value}`));
    featureNames.push(`${field}=__OTHER__`);
  });

  return { fields, valueSets, featureNames };
}

function productFactorKeys(sample) {
  return {
    categoryHistory: sample.category,
    colorHistory: sample.color,
    fabricHistory: sample.fabric,
    categoryColorHistory: `${sample.category}||${sample.color}`,
    categoryFabricHistory: `${sample.category}||${sample.fabric}`,
    colorFabricHistory: `${sample.color}||${sample.fabric}`,
    categoryColorFabricHistory: `${sample.category}||${sample.color}||${sample.fabric}`,
  };
}

function buildHorizonModelSpec(baseSpec, trainSamples, targetField) {
  const factorNames = [
    "categoryHistory",
    "colorHistory",
    "fabricHistory",
    "categoryColorHistory",
    "categoryFabricHistory",
    "colorFabricHistory",
    "categoryColorFabricHistory",
  ];
  const globalMean = trainSamples.reduce((sum, sample) => sum + sample[targetField], 0) / Math.max(1, trainSamples.length);
  const factorStats = Object.fromEntries(factorNames.map((name) => [name, new Map()]));

  trainSamples.forEach((sample) => {
    const keys = productFactorKeys(sample);
    factorNames.forEach((name) => {
      const key = keys[name];
      const current = factorStats[name].get(key) || { sum: 0, count: 0 };
      current.sum += sample[targetField];
      current.count += 1;
      factorStats[name].set(key, current);
    });
  });

  return {
    ...baseSpec,
    featureNames: [
      ...baseSpec.featureNames,
      ...factorNames.map((name) => `${name}_demand_ratio`),
      ...factorNames.map((name) => `${name}_sample_count_log`),
    ],
    productFactors: {
      names: factorNames,
      globalMean,
      priorWeight: 20,
      stats: factorStats,
    },
  };
}

function productFactorVector(sample, modelSpec) {
  if (!modelSpec.productFactors) return [];

  const keys = productFactorKeys(sample);
  const ratios = [];
  const counts = [];

  modelSpec.productFactors.names.forEach((name) => {
    const stat = modelSpec.productFactors.stats[name].get(keys[name]) || { sum: 0, count: 0 };
    const smoothedMean = (
      stat.sum + modelSpec.productFactors.globalMean * modelSpec.productFactors.priorWeight
    ) / (stat.count + modelSpec.productFactors.priorWeight);
    ratios.push(smoothedMean / Math.max(1, modelSpec.productFactors.globalMean));
    counts.push(Math.log1p(stat.count));
  });

  return [...ratios, ...counts];
}

function vectorize(sample, modelSpec) {
  const vector = [
    1,
    (sample.categoryTrend || 0) / 100,
    (sample.colorTrend || 0) / 100,
    (sample.fabricTrend || 0) / 100,
    (sample.trendComposite || 0) / 100,
    Math.min(3, Math.max(0, sample.trendMomentum || 1)),
    (sample.trendCoverage || 0) / 3,
  ];

  modelSpec.fields.forEach((field) => {
    const values = modelSpec.valueSets[field];
    values.forEach((value) => vector.push(sample[field] === value ? 1 : 0));
    vector.push(values.has(sample[field]) ? 0 : 1);
  });

  vector.push(...productFactorVector(sample, modelSpec));

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

function trainRidge(samples, modelSpec, targetField, lambda, targetMode) {
  const p = modelSpec.featureNames.length;
  const xtx = Array.from({ length: p }, () => Array(p).fill(0));
  const xty = Array(p).fill(0);

  samples.forEach((sample) => {
    const x = vectorize(sample, modelSpec);
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

function predict(sample, modelSpec, weights, targetMode, calibration = 1) {
  const x = vectorize(sample, modelSpec);
  const rawPrediction = x.reduce((sum, value, index) => sum + value * weights[index], 0);
  const prediction = targetMode === "log1p" ? Math.expm1(rawPrediction) : targetMode === "sqrt" ? rawPrediction ** 2 : rawPrediction;
  return Math.max(0, prediction * calibration);
}

function calibrationFactor(samples, modelSpec, weights, targetField, targetMode) {
  const actualTotal = samples.reduce((sum, sample) => sum + sample[targetField], 0);
  const forecastTotal = samples.reduce((sum, sample) => (
    sum + predict(sample, modelSpec, weights, targetMode)
  ), 0);
  return forecastTotal > 0 ? actualTotal / forecastTotal : 1;
}

function metrics(samples, modelSpec, weights, targetField, targetMode, calibration = 1) {
  let absoluteError = 0;
  let squaredError = 0;
  let signedError = 0;
  let actualTotal = 0;
  let forecastTotal = 0;
  let mapeSum = 0;
  let mapeCount = 0;

  samples.forEach((sample) => {
    const forecast = predict(sample, modelSpec, weights, targetMode, calibration);
    const actual = sample[targetField];
    const error = forecast - actual;
    absoluteError += Math.abs(error);
    squaredError += error ** 2;
    signedError += error;
    actualTotal += actual;
    forecastTotal += forecast;
    if (actual > 0) {
      mapeSum += Math.abs(error) / actual;
      mapeCount += 1;
    }
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
    mapePct: Number(((mapeSum / Math.max(1, mapeCount)) * 100).toFixed(2)),
  };
}

function meanMetric(samples, targetField) {
  return samples.reduce((sum, sample) => sum + sample[targetField], 0) / Math.max(1, samples.length);
}

function buildGroupStats(samples, targetField) {
  const factorNames = [
    "categoryHistory",
    "colorHistory",
    "fabricHistory",
    "categoryColorHistory",
    "categoryFabricHistory",
    "colorFabricHistory",
    "categoryColorFabricHistory",
  ];
  const stats = Object.fromEntries(factorNames.map((name) => [name, new Map()]));
  const globalMean = meanMetric(samples, targetField);

  samples.forEach((sample) => {
    const keys = productFactorKeys(sample);
    factorNames.forEach((name) => {
      const current = stats[name].get(keys[name]) || { sum: 0, count: 0 };
      current.sum += sample[targetField];
      current.count += 1;
      stats[name].set(keys[name], current);
    });
  });

  return { factorNames, globalMean, stats };
}

function groupMean(groupStats, factorName, key, priorWeight) {
  const stat = groupStats.stats[factorName].get(key) || { sum: 0, count: 0 };
  return {
    mean: (stat.sum + groupStats.globalMean * priorWeight) / (stat.count + priorWeight),
    count: stat.count,
  };
}

function hierarchicalPredict(sample, groupStats, config, calibration = 1) {
  const keys = productFactorKeys(sample);
  const exact = groupMean(groupStats, "categoryColorFabricHistory", keys.categoryColorFabricHistory, config.priorWeight);
  const categoryColor = groupMean(groupStats, "categoryColorHistory", keys.categoryColorHistory, config.priorWeight);
  const categoryFabric = groupMean(groupStats, "categoryFabricHistory", keys.categoryFabricHistory, config.priorWeight);
  const colorFabric = groupMean(groupStats, "colorFabricHistory", keys.colorFabricHistory, config.priorWeight);
  const category = groupMean(groupStats, "categoryHistory", keys.categoryHistory, config.priorWeight);
  const color = groupMean(groupStats, "colorHistory", keys.colorHistory, config.priorWeight);
  const fabric = groupMean(groupStats, "fabricHistory", keys.fabricHistory, config.priorWeight);

  const components = [
    { ...exact, baseWeight: config.exactWeight },
    { ...categoryColor, baseWeight: config.categoryColorWeight },
    { ...categoryFabric, baseWeight: config.categoryFabricWeight },
    { ...colorFabric, baseWeight: config.colorFabricWeight },
    { ...category, baseWeight: config.categoryWeight },
    { ...color, baseWeight: config.colorWeight },
    { ...fabric, baseWeight: config.fabricWeight },
    { mean: groupStats.globalMean, count: 1, baseWeight: config.globalWeight },
  ];

  const weighted = components.map((component) => ({
    value: component.mean,
    weight: component.baseWeight * Math.log1p(component.count + config.countBoost),
  })).filter((component) => component.weight > 0);
  const weightTotal = weighted.reduce((sum, component) => sum + component.weight, 0);
  const prediction = weightTotal
    ? weighted.reduce((sum, component) => sum + component.value * component.weight, 0) / weightTotal
    : groupStats.globalMean;

  return Math.max(0, prediction * calibration);
}

function hierarchicalCalibrationFactor(samples, groupStats, targetField, config) {
  const actualTotal = samples.reduce((sum, sample) => sum + sample[targetField], 0);
  const forecastTotal = samples.reduce((sum, sample) => sum + hierarchicalPredict(sample, groupStats, config), 0);
  return forecastTotal > 0 ? actualTotal / forecastTotal : 1;
}

function hierarchicalMetrics(samples, groupStats, targetField, config, calibration = 1) {
  let absoluteError = 0;
  let squaredError = 0;
  let signedError = 0;
  let actualTotal = 0;
  let forecastTotal = 0;
  let mapeSum = 0;
  let mapeCount = 0;

  samples.forEach((sample) => {
    const forecast = hierarchicalPredict(sample, groupStats, config, calibration);
    const actual = sample[targetField];
    const error = forecast - actual;
    absoluteError += Math.abs(error);
    squaredError += error ** 2;
    signedError += error;
    actualTotal += actual;
    forecastTotal += forecast;
    if (actual > 0) {
      mapeSum += Math.abs(error) / actual;
      mapeCount += 1;
    }
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
    mapePct: Number(((mapeSum / Math.max(1, mapeCount)) * 100).toFixed(2)),
  };
}

function hierarchicalConfigs() {
  const priorWeights = [5, 10, 20, 40];
  const countBoosts = [0, 3, 10];
  const weightSets = [
    {
      name: "exact-heavy",
      exactWeight: 4,
      categoryColorWeight: 2.2,
      categoryFabricWeight: 1.4,
      colorFabricWeight: 1,
      categoryWeight: 1.2,
      colorWeight: 0.8,
      fabricWeight: 0.5,
      globalWeight: 0.2,
    },
    {
      name: "color-heavy",
      exactWeight: 3.5,
      categoryColorWeight: 2.6,
      categoryFabricWeight: 1.1,
      colorFabricWeight: 1.3,
      categoryWeight: 1,
      colorWeight: 1,
      fabricWeight: 0.4,
      globalWeight: 0.2,
    },
    {
      name: "balanced",
      exactWeight: 3,
      categoryColorWeight: 1.8,
      categoryFabricWeight: 1.5,
      colorFabricWeight: 1.1,
      categoryWeight: 1.2,
      colorWeight: 0.7,
      fabricWeight: 0.7,
      globalWeight: 0.3,
    },
  ];

  return weightSets.flatMap((weights) => priorWeights.flatMap((priorWeight) => (
    countBoosts.map((countBoost) => ({ ...weights, priorWeight, countBoost }))
  )));
}

function trainHorizonModel({ horizon, trainSamples, validationSamples, testSamples, baseModelSpec }) {
  const targetField = `future${horizon}wDemand`;
  const modelSpec = buildHorizonModelSpec(baseModelSpec, trainSamples, targetField);
  const lambdas = [0.1, 1, 5, 10, 25, 50, 100, 250, 500];
  const targetModes = ["log1p", "sqrt", "raw"];

  const ridgeCandidates = targetModes.flatMap((targetMode) => lambdas.flatMap((lambda) => {
    const weights = trainRidge(trainSamples, modelSpec, targetField, lambda, targetMode);
    const validationCalibration = calibrationFactor(validationSamples, modelSpec, weights, targetField, targetMode);
    const validationRaw = metrics(validationSamples, modelSpec, weights, targetField, targetMode);
    return [
      {
        targetMode,
        modelType: "ridge",
        lambda,
        calibrated: false,
        calibration: 1,
        weights,
        validationRaw,
        validation: validationRaw,
      },
      {
        targetMode,
        modelType: "ridge",
        lambda,
        calibrated: true,
        calibration: validationCalibration,
        weights,
        validationRaw,
        validation: metrics(validationSamples, modelSpec, weights, targetField, targetMode, validationCalibration),
      },
    ];
  }));
  const groupStats = buildGroupStats(trainSamples, targetField);
  const hierarchicalCandidates = hierarchicalConfigs().flatMap((config) => {
    const validationCalibration = hierarchicalCalibrationFactor(validationSamples, groupStats, targetField, config);
    const validationRaw = hierarchicalMetrics(validationSamples, groupStats, targetField, config);
    return [
      {
        modelType: "hierarchical-benchmark",
        config,
        calibrated: false,
        calibration: 1,
        groupStats,
        validationRaw,
        validation: validationRaw,
      },
      {
        modelType: "hierarchical-benchmark",
        config,
        calibrated: true,
        calibration: validationCalibration,
        groupStats,
        validationRaw,
        validation: hierarchicalMetrics(validationSamples, groupStats, targetField, config, validationCalibration),
      },
    ];
  });
  const candidates = [...ridgeCandidates, ...hierarchicalCandidates];

  const best = candidates.reduce((winner, candidate) => (
    candidate.validation.wape < winner.validation.wape ? candidate : winner
  ), candidates[0]);
  const selectedModel = best.modelType === "ridge"
    ? {
      type: "Ridge regression",
      targetMode: best.targetMode,
      lambda: best.lambda,
      calibratedOnValidation: best.calibrated,
      validationCalibrationFactor: Number(best.calibration.toFixed(4)),
    }
    : {
      type: "Hierarchical benchmark ensemble",
      configName: best.config.name,
      priorWeight: best.config.priorWeight,
      countBoost: best.config.countBoost,
      calibratedOnValidation: best.calibrated,
      validationCalibrationFactor: Number(best.calibration.toFixed(4)),
    };
  const split = best.modelType === "ridge"
    ? {
      train: metrics(trainSamples, modelSpec, best.weights, targetField, best.targetMode, best.calibration),
      validation: best.validation,
      test: metrics(testSamples, modelSpec, best.weights, targetField, best.targetMode, best.calibration),
    }
    : {
      train: hierarchicalMetrics(trainSamples, best.groupStats, targetField, best.config, best.calibration),
      validation: best.validation,
      test: hierarchicalMetrics(testSamples, best.groupStats, targetField, best.config, best.calibration),
    };

  return {
    horizonWeeks: horizon,
    target: `total demand from W0-W${horizon - 1}`,
    selectedModel,
    productFactorFeatures: modelSpec.productFactors.names.map((name) => `${name}_demand_ratio / ${name}_sample_count_log`),
    split,
    validationTuning: candidates.map((candidate) => ({
      modelType: candidate.modelType,
      targetMode: candidate.targetMode,
      lambda: candidate.lambda,
      configName: candidate.config?.name,
      priorWeight: candidate.config?.priorWeight,
      countBoost: candidate.config?.countBoost,
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
const trendRows = parseTrendRows(mysqlTsv(trendsQuery));
const samples = splitSamples(productSamples(rows, trendRows));
const trainSamples = samples.filter((sample) => sample.split === "train");
const validationSamples = samples.filter((sample) => sample.split === "validation");
const testSamples = samples.filter((sample) => sample.split === "test");
const baseModelSpec = buildVocabulary(trainSamples);
const horizonModels = HORIZONS.map((horizon) => trainHorizonModel({
  horizon,
  trainSamples,
  validationSamples,
  testSamples,
  baseModelSpec,
}));

const report = {
  generatedAt: new Date().toISOString(),
  source: {
    database: DB_NAME,
    table: "product_weekly_lifecycle",
    rawRowsUsed: rows.length,
    productSamples: samples.length,
  },
  problemDefinition: {
    inputFeatures: ["category", "color", "fabric", "Google Trends signals derived from category/color/fabric"],
    internalProductFactors: [
      "historical category demand benchmark",
      "historical color demand benchmark",
      "historical fabric demand benchmark",
      "historical category-color demand benchmark",
      "historical category-fabric demand benchmark",
      "historical color-fabric demand benchmark",
      "historical category-color-fabric demand benchmark",
      "training sample count for each benchmark, log-scaled",
      "recent 8-week Google Trends level for category, color, and fabric before release",
      "weighted Google Trends composite: category 50%, color 25%, fabric 25%",
      "Google Trends momentum versus the previous 8 weeks",
    ],
    outputs: [
      "future 4-week total demand: W0-W3 units",
      "future 8-week total demand: W0-W7 units",
      "future 12-week total demand: W0-W11 units",
    ],
    note: "The user-facing product inputs are category, color, and fabric. Product-related factors are derived from those inputs using train-split historical demand benchmarks. Google Trends features use only trend observations on or before product release date, so validation/test products do not leak future demand into feature generation. The model does not use actual early sales, price, discount, stock, store coverage, or weather.",
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
    baseFeatures: baseModelSpec.featureNames,
    productFactorFeatures: horizonModels[0]?.productFactorFeatures || [],
  },
  horizons: horizonModels,
};

mkdirSync(dirname(resolve(OUTPUT)), { recursive: true });
writeFileSync(OUTPUT, `${JSON.stringify(report, null, 2)}\n`);

console.log(`Wrote ${OUTPUT}`);
horizonModels.forEach((model) => {
  console.log(`${model.horizonWeeks}w test WAPE: ${model.split.test.wapePct}%`);
  console.log(`${model.horizonWeeks}w test accuracy: ${model.split.test.accuracyPct}%`);
  console.log(`${model.horizonWeeks}w test bias: ${model.split.test.biasPct}%`);
});
