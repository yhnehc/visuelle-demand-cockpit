import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AlertTriangle,
  Bell,
  ArrowRightLeft,
  ChevronRight,
  Clock3,
  HelpCircle,
  LineChart,
  PackageCheck,
  ShoppingCart,
  Search,
  X,
  Store,
  TrendingUp,
} from "lucide-react";
import "./styles.css";

const EXISTING_FORECAST_HORIZON = 12;
const MAX_LIFECYCLE_WEEK = 52;
const MIN_CREDIBLE_ANALOGS = 3;
const LM_STUDIO_BASE_URL = "/lmstudio";

const actionIcon = {
  Restock: PackageCheck,
  Reallocate: ArrowRightLeft,
  Watch: Clock3,
  "Markdown Review": AlertTriangle,
};

function formatK(value) {
  if (value >= 1000) return `${(value / 1000).toFixed(value >= 10000 ? 0 : 1)}K`;
  return `${Math.round(value)}`;
}

function formatDate(value) {
  return new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", year: "numeric" }).format(new Date(value));
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

function closestOption(value, options) {
  if (!value) return null;
  const normalized = String(value).trim().toLowerCase();
  return options.find((option) => option.toLowerCase() === normalized)
    || options.find((option) => normalized.includes(option.toLowerCase()) || option.toLowerCase().includes(normalized))
    || null;
}

function parseModelJson(text) {
  const cleaned = String(text || "").replace(/```json|```/g, "").trim();
  const match = cleaned.match(/\{[\s\S]*\}/);
  if (!match) throw new Error("LM Studio did not return JSON.");
  return JSON.parse(match[0]);
}

async function getLmStudioModelId() {
  try {
    const response = await fetch(`${LM_STUDIO_BASE_URL}/models`);
    if (!response.ok) throw new Error("models unavailable");
    const payload = await response.json();
    return payload.data?.[0]?.id || "local-model";
  } catch {
    return "local-model";
  }
}

async function classifyProductImageWithLmStudio({ imageDataUrl, options }) {
  const model = await getLmStudioModelId();
  const response = await fetch(`${LM_STUDIO_BASE_URL}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      temperature: 0,
      messages: [
        {
          role: "system",
          content: "You classify fashion product images for demand planning. Return only valid JSON. Choose values only from the provided allowed lists. If uncertain, choose the closest available option.",
        },
        {
          role: "user",
          content: [
            {
              type: "text",
              text: `Analyze this fashion product image and infer the best matching attributes.\nAllowed categories: ${options.categories.join(", ")}\nAllowed colors: ${options.colors.join(", ")}\nAllowed fabrics: ${options.fabrics.join(", ")}\nReturn JSON only with keys: category, color, fabric, reasoning.`,
            },
            { type: "image_url", image_url: { url: imageDataUrl } },
          ],
        },
      ],
    }),
  });

  if (!response.ok) {
    throw new Error(`LM Studio request failed (${response.status}). Check that LM Studio is running on 127.0.0.1:1234 with a vision-capable model loaded.`);
  }

  const payload = await response.json();
  const content = payload.choices?.[0]?.message?.content;
  const parsed = parseModelJson(content);
  const category = closestOption(parsed.category, options.categories);
  const color = closestOption(parsed.color, options.colors);
  const fabric = closestOption(parsed.fabric, options.fabrics);

  if (!category || !color || !fabric) {
    throw new Error("LM Studio returned attributes that could not be mapped to the Visuelle options.");
  }

  return {
    category,
    color,
    fabric,
    reasoning: parsed.reasoning || "Matched from uploaded product image.",
  };
}

function confidenceLabel(score) {
  if (score >= 72) return "High";
  if (score >= 55) return "Medium";
  return "Low";
}

function calculateEvidenceConfidence(product) {
  const analogTotals = product.analogs?.map((item) => launchTotal(item)).filter((value) => Number.isFinite(value)) || [];
  const analogMedian = quantile(analogTotals, 0.5);
  const analogIqr = quantile(analogTotals, 0.75) - quantile(analogTotals, 0.25);
  const consistency = analogMedian ? Math.max(0, 1 - Math.min(1.2, analogIqr / analogMedian) / 1.2) : 0;
  const actualHistoryScore = lifecycleSales(product).length >= 12 ? 25 : 0;
  const analogCoverageScore = Math.min(20, analogTotals.length * 4);
  const consistencyScore = consistency * 25;
  const discountScore = product.avgDiscount <= 0.1 ? 15 : product.avgDiscount <= 0.2 ? 11 : 7;
  const storeCoverageScore = product.stores >= 80 ? 15 : product.stores >= 50 ? 11 : 7;

  return Math.max(35, Math.min(92, Math.round(
    actualHistoryScore + analogCoverageScore + consistencyScore + discountScore + storeCoverageScore
  )));
}

function calculateNewProductConfidence(analogs) {
  const totals = analogs.map((item) => lifecycleTotal(item)).filter((value) => Number.isFinite(value));
  const analogMedian = quantile(totals, 0.5);
  const analogIqr = quantile(totals, 0.75) - quantile(totals, 0.25);
  const consistency = analogMedian ? Math.max(0, 1 - Math.min(1.2, analogIqr / analogMedian) / 1.2) : 0;
  if (totals.length < 3) {
    return Math.max(25, Math.min(45, Math.round(20 + totals.length * 8 + consistency * 10)));
  }
  const analogCoverageScore = Math.min(40, totals.length * 8);
  const consistencyScore = consistency * 35;

  return Math.max(25, Math.min(85, Math.round(20 + analogCoverageScore + consistencyScore)));
}

function classForSignal(signal) {
  if (signal.includes("Restock") || signal.includes("Allocation")) return "positive";
  if (signal.includes("Hold") || signal.includes("transfer") || signal.includes("margin")) return "risk";
  return "neutral";
}

function analogMatchScore(baseProduct, analog) {
  if (!baseProduct) return analog.matchScore ? Math.min(100, Math.round(analog.matchScore * 10)) : 0;
  return (
    (analog.category === baseProduct.category ? 40 : 0) +
    (analog.color === baseProduct.color ? 20 : 0) +
    (analog.fabric === baseProduct.fabric ? 25 : 0) +
    (analog.season === baseProduct.season ? 15 : 0)
  );
}

function categoryBenchmarks(data, product) {
  const categoryTotals = data.newProductModel.catalog
    .filter((item) => item.category === product.category)
    .map((item) => launchTotal(item))
    .filter((value) => Number.isFinite(value));

  return {
    median: Math.round(quantile(categoryTotals, 0.5)),
    topQuartile: Math.round(quantile(categoryTotals, 0.75)),
    count: categoryTotals.length,
  };
}

function sumWeeks(values, weeks) {
  return weeks.reduce((sum, week) => sum + (values[week] || 0), 0);
}

function lifecycleSales(item) {
  return item?.actualLifecycle?.length ? item.actualLifecycle : (item?.weekSales || []);
}

function lifecycleWindow(values) {
  return values.length ? `W0-W${values.length - 1}` : "No actuals";
}

function lifecyclePoints(values) {
  return values.map((value, week) => ({ week, value }));
}

function launchSales(item) {
  const source = lifecycleSales(item).length ? lifecycleSales(item) : (item?.weekSales || []);
  return Array.from({ length: 12 }, (_, week) => source[week] || 0);
}

function launchTotal(item) {
  return launchSales(item).reduce((sum, value) => sum + value, 0);
}

function lifecycleTotal(item) {
  return lifecycleSales(item).reduce((sum, value) => sum + value, 0);
}

function analogRangeByWeek(analogs, labels) {
  const top = [];
  const bottom = [];

  labels.forEach((week) => {
    const values = analogs
      .map((item) => lifecycleSales(item)[week])
      .filter((value) => Number.isFinite(value));

    if (!values.length) return;
    top.push({ week, value: Math.round(quantile(values, 0.75)) });
    bottom.push({ week, value: Math.round(quantile(values, 0.25)) });
  });

  return [top, bottom];
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function weightedAverage(values) {
  const totalWeight = values.reduce((sum, item) => sum + item.weight, 0);
  if (!totalWeight) return 0;
  return values.reduce((sum, item) => sum + item.value * item.weight, 0) / totalWeight;
}

function calculateExistingAnalogForecast(product, matchedAnalogs, catalog) {
  const actualLifecycle = lifecycleSales(product);
  const actualEndWeek = actualLifecycle.length - 1;
  const forecastWeeks = Array.from(
    { length: Math.min(EXISTING_FORECAST_HORIZON, Math.max(0, MAX_LIFECYCLE_WEEK - actualEndWeek)) },
    (_, index) => actualEndWeek + index + 1
  );

  const actualTotal = actualLifecycle.reduce((sum, value) => sum + value, 0);
  const trendSignal = product.googleTrendSignal || {};
  const trendAdjustment = Number.isFinite(trendSignal.adjustment) ? trendSignal.adjustment : 1;
  const recentStart = Math.max(0, actualLifecycle.length - 4);
  const actualRecent = actualLifecycle.slice(recentStart).reduce((sum, value) => sum + value, 0);
  const matchedIds = new Set(matchedAnalogs.map((item) => item.id));
  const candidates = catalog
    .filter((item) => item.id !== product.id && item.category === product.category)
    .map((item) => {
      const candidateLifecycle = lifecycleSales(item);
      const candidateActualWindow = candidateLifecycle.slice(0, actualLifecycle.length);
      const candidateTotal = candidateActualWindow.reduce((sum, value) => sum + value, 0);
      const candidateRecent = candidateLifecycle.slice(recentStart, actualLifecycle.length).reduce((sum, value) => sum + value, 0);
      if (!candidateTotal || !candidateLifecycle.length) return null;

      const totalScale = actualTotal / candidateTotal;
      const recentScale = candidateRecent ? actualRecent / candidateRecent : totalScale;
      const scale = clamp((totalScale * 0.65) + (recentScale * 0.35), 0.25, 3);
      const matchScore = analogMatchScore(product, item);

      return {
        item,
        lifecycle: candidateLifecycle,
        scale,
        matchScore,
        isMatched: matchedIds.has(item.id),
        weight: matchedIds.has(item.id) ? Math.max(0.75, matchScore / 100) : 0.35,
      };
    })
    .filter(Boolean);

  const points = [];
  const top = [];
  const bottom = [];
  const weeklyCoverage = [];
  let baseTotal = 0;
  const candidateTotals = candidates
    .map((candidate) => {
      const values = forecastWeeks
        .map((week) => candidate.lifecycle[week])
        .filter((value) => Number.isFinite(value));
      return values.length ? values.reduce((sum, value) => sum + value * candidate.scale, 0) : null;
    })
    .filter((value) => Number.isFinite(value));

  let lastBaseForecast = Math.max(0, actualLifecycle[actualLifecycle.length - 1] || 0);

  forecastWeeks.forEach((week, index) => {
    const matchedValues = candidates
      .filter((candidate) => candidate.isMatched && Number.isFinite(candidate.lifecycle[week]))
      .map((candidate) => ({ value: candidate.lifecycle[week] * candidate.scale, weight: candidate.weight }));
    const categoryValues = candidates
      .filter((candidate) => Number.isFinite(candidate.lifecycle[week]))
      .map((candidate) => ({ value: candidate.lifecycle[week] * candidate.scale, weight: candidate.weight }));
    const values = matchedValues.length >= 3 ? matchedValues : categoryValues;
    const baseForecastValue = values.length ? weightedAverage(values) : lastBaseForecast * 0.82;
    const roundedValue = Math.max(0, Math.round(baseForecastValue * trendAdjustment));
    baseTotal += Math.max(0, Math.round(baseForecastValue));

    points.push({ week, value: roundedValue });
    if (values.length) {
      top.push({ week, value: Math.round(quantile(values.map((item) => item.value), 0.75) * trendAdjustment) });
      bottom.push({ week, value: Math.round(quantile(values.map((item) => item.value), 0.25) * trendAdjustment) });
    } else {
      top.push({ week, value: roundedValue });
      bottom.push({ week, value: roundedValue });
    }
    weeklyCoverage.push({ week, matched: matchedValues.length, category: categoryValues.length });
    lastBaseForecast = Math.max(0, baseForecastValue);
  });

  const total = points.reduce((sum, point) => sum + point.value, 0);
  const matchedCoverageWeeks = weeklyCoverage.filter((item) => item.matched >= 3).length;
  const averageCategoryCoverage = weeklyCoverage.length
    ? weeklyCoverage.reduce((sum, item) => sum + Math.min(1, item.category / 3), 0) / weeklyCoverage.length
    : 0;
  const rawCandidateMedian = quantile(candidateTotals, 0.5);
  const candidateMedian = Math.round(rawCandidateMedian * trendAdjustment);
  const candidateTopQuartile = Math.round(quantile(candidateTotals, 0.75) * trendAdjustment);
  const candidateIqr = quantile(candidateTotals, 0.75) - quantile(candidateTotals, 0.25);
  const consistency = rawCandidateMedian ? Math.max(0, 1 - Math.min(1.25, candidateIqr / rawCandidateMedian) / 1.25) : 0;
  const matchedAnalogScore = Math.min(10, matchedAnalogs.length * 2);
  const matchedTailScore = (matchedCoverageWeeks / Math.max(1, forecastWeeks.length)) * 35;
  const categoryFallbackScore = averageCategoryCoverage * 10;
  const trendCoverage = trendSignal.coverageWeeks || {};
  const trendCoverageScore = trendCoverage.recent >= 8 && trendCoverage.forecast >= 8 ? 5 : 0;
  const confidenceScore = clamp(Math.round(
    10 +
    Math.min(10, (actualLifecycle.length / 12) * 10) +
    matchedAnalogScore +
    matchedTailScore +
    categoryFallbackScore +
    consistency * 15 +
    trendCoverageScore
  ), 25, 88);

  return {
    points,
    top,
    bottom,
    total,
    baseTotal,
    window: forecastWeeks.length ? `W${forecastWeeks[0]}-W${forecastWeeks.at(-1)}` : "No forecast",
    trendAdjustment,
    trendSignal,
    weeklyCoverage,
    matchedCoverageWeeks,
    candidateCount: candidates.length,
    candidateMedian,
    candidateTopQuartile,
    confidenceScore,
    confidenceBreakdown: {
      actualHistory: Math.round(Math.min(10, (actualLifecycle.length / 12) * 10)),
      matchedAnalogs: Math.round(matchedAnalogScore),
      matchedTail: Math.round(matchedTailScore),
      categoryFallback: Math.round(categoryFallbackScore),
      consistency: Math.round(consistency * 15),
      trendCoverage: trendCoverageScore,
    },
  };
}

function valueForModelFeature(feature, product, modelSpec, horizonModel) {
  const actual = lifecycleSales(product);
  const recent = actual.slice(-4);
  const padded = [...Array(Math.max(0, 4 - recent.length)).fill(0), ...recent];
  const recentTotal = Math.max(1, padded.reduce((sum, value) => sum + value, 0));
  const first2 = padded.slice(0, 2).reduce((sum, value) => sum + value, 0);
  const last2 = padded.slice(2, 4).reduce((sum, value) => sum + value, 0);
  const trend = product.googleTrendSignal || {};
  const trendValue = Number.isFinite(trend.forecastAvg) ? trend.forecastAvg : Number.isFinite(trend.recentAvg) ? trend.recentAvg : 0;

  if (feature === "intercept") return 1;
  if (feature === "log_observed4w_units") return Math.log1p(recentTotal);
  if (feature === "log_observed_first2w_units") return Math.log1p(first2);
  if (feature === "log_observed_last2w_units") return Math.log1p(last2);
  if (feature === "observed_trend_ratio_capped") return Math.min(5, Math.max(0, first2 > 0 ? last2 / first2 : 1));
  if (feature === "week0_share") return padded[0] / recentTotal;
  if (feature === "week1_share") return padded[1] / recentTotal;
  if (feature === "week2_share") return padded[2] / recentTotal;
  if (feature === "week3_share") return padded[3] / recentTotal;
  if (feature === "avg_price") return product.price || 0;
  if (feature === "avg_discount_w0_w3") return product.avgDiscount || 0;
  if (feature === "max_discount_w0_w3") return product.avgDiscount || 0;
  if (feature === "log_store_coverage") return Math.log1p(product.stores || 0);
  if (feature === "log_restock_signal") return Math.log1p(product.restockEvents || product.restockQty || 0);
  if (feature === "log_restock_store_rows") return Math.log1p(product.restockEvents || 0);
  if (feature === "category_trend" || feature === "color_trend" || feature === "fabric_trend" || feature === "combined_trend") return trendValue / 100;
  if (feature === "category_benchmark_ratio" || feature === "category_benchmark_count_log") {
    const stat = horizonModel.categoryBenchmark?.stats?.[product.category] || { sum: 0, count: 0 };
    const globalMean = horizonModel.categoryBenchmark?.globalMean || 1;
    const priorWeight = horizonModel.categoryBenchmark?.priorWeight || 20;
    const benchmark = (stat.sum + globalMean * priorWeight) / (stat.count + priorWeight);
    return feature === "category_benchmark_ratio" ? benchmark / Math.max(1, globalMean) : Math.log1p(stat.count);
  }
  if (feature.startsWith("category=")) {
    const value = feature.slice("category=".length);
    const known = modelSpec.valueSets?.category || [];
    return value === "__OTHER__" ? (known.includes(product.category) ? 0 : 1) : product.category === value ? 1 : 0;
  }
  if (feature.startsWith("color=")) {
    const value = feature.slice("color=".length);
    const known = modelSpec.valueSets?.color || [];
    return value === "__OTHER__" ? (known.includes(product.color) ? 0 : 1) : product.color === value ? 1 : 0;
  }
  if (feature.startsWith("fabric=")) {
    const value = feature.slice("fabric=".length);
    const known = modelSpec.valueSets?.fabric || [];
    return value === "__OTHER__" ? (known.includes(product.fabric) ? 0 : 1) : product.fabric === value ? 1 : 0;
  }
  return 0;
}

function predictExistingDemand(product, existingModel, horizon) {
  const productPredictions = existingModel?.predictionsByProductId?.[String(product.id)];
  const directPrediction = productPredictions?.[`next${horizon}wDemand`];
  if (Number.isFinite(directPrediction)) return directPrediction;

  const horizonModel = existingModel?.horizons?.[String(horizon)];
  if (!existingModel?.baseSpec || !horizonModel) return null;
  const raw = horizonModel.featureNames.reduce((sum, feature, index) => (
    sum + valueForModelFeature(feature, product, existingModel.baseSpec, horizonModel) * horizonModel.weights[index]
  ), 0);
  const transformed = horizonModel.targetMode === "log1p"
    ? Math.expm1(raw)
    : horizonModel.targetMode === "sqrt"
      ? raw ** 2
      : raw;
  return Math.max(0, transformed * (horizonModel.calibration || 1));
}

function existingModelPredictions(product, existingModel) {
  const next4w = predictExistingDemand(product, existingModel, 4);
  const next8wRaw = predictExistingDemand(product, existingModel, 8);
  const next12wRaw = predictExistingDemand(product, existingModel, 12);
  if (![next4w, next8wRaw, next12wRaw].every(Number.isFinite)) {
    return { next4w: null, next8w: null, next12w: null };
  }
  const next8w = Math.max(next4w, next8wRaw);
  const next12w = Math.max(next8w, next12wRaw);
  return { next4w, next8w, next12w };
}

function existingModelCategoryBenchmark(product, existingModel, horizon) {
  const horizonModel = existingModel?.horizons?.[String(horizon)];
  const stat = horizonModel?.categoryBenchmark?.stats?.[product.category];
  const globalMean = horizonModel?.categoryBenchmark?.globalMean;
  const priorWeight = horizonModel?.categoryBenchmark?.priorWeight || 20;
  if (!stat || !Number.isFinite(globalMean)) return null;
  return (stat.sum + globalMean * priorWeight) / (stat.count + priorWeight);
}

function calculateExistingProductForecast(product, matchedAnalogs, catalog, existingModel) {
  const analogForecast = calculateExistingAnalogForecast(product, matchedAnalogs, catalog);
  const modelPredictions = existingModelPredictions(product, existingModel);
  const modelTotal12 = modelPredictions.next12w;
  if (!Number.isFinite(modelTotal12)) {
    return {
      ...analogForecast,
      points: [],
      top: [],
      bottom: [],
      total: 0,
      baseTotal: 0,
      window: "Model unavailable",
      source: "ML model unavailable",
      modelPredictions: {
        ...modelPredictions,
        categoryBenchmarkNext12: existingModelCategoryBenchmark(product, existingModel, 12),
      },
      confidenceScore: 0,
    };
  }

  const analogTotal = analogForecast.total || 1;
  const scale = modelTotal12 / analogTotal;
  const points = analogForecast.points.map((point) => ({ ...point, value: Math.max(0, Math.round(point.value * scale)) }));
  const top = analogForecast.top.map((point) => ({ ...point, value: Math.max(0, Math.round(point.value * scale)) }));
  const bottom = analogForecast.bottom.map((point) => ({ ...point, value: Math.max(0, Math.round(point.value * scale)) }));
  const total = points.reduce((sum, point) => sum + point.value, 0);
  const validation = existingModel.validationSummary?.["12"];

  return {
    ...analogForecast,
    points,
    top,
    bottom,
    total,
    baseTotal: Math.round(modelTotal12),
    source: "ML model",
    modelPredictions: {
      ...modelPredictions,
      categoryBenchmarkNext12: existingModelCategoryBenchmark(product, existingModel, 12),
      testAccuracyPct: validation?.testAccuracyPct,
      testWapePct: validation?.testWapePct,
      testBiasPct: validation?.testBiasPct,
    },
    confidenceScore: validation?.testAccuracyPct ? Math.round(validation.testAccuracyPct) : analogForecast.confidenceScore,
  };
}

function decideExistingProductAction({ forecast, avgDiscount, categoryBenchmark }) {
  if (!forecast.points.length || forecast.confidenceScore < 50) return "Watch";
  if (avgDiscount >= 0.2) return "Markdown Review";
  if (forecast.source === "ML model") {
    const modelBenchmark = forecast.modelPredictions?.categoryBenchmarkNext12 || categoryBenchmark?.median || 0;
    if (forecast.total >= modelBenchmark * 1.25 && forecast.confidenceScore >= 70) return "Restock";
    if (forecast.total < Math.round(modelBenchmark * 0.75)) return "Watch";
    return "Reallocate";
  }
  if (forecast.total >= forecast.candidateTopQuartile && forecast.confidenceScore >= 70 && forecast.matchedCoverageWeeks >= 6) return "Restock";
  if (forecast.total < Math.round(forecast.candidateMedian * 0.75)) return "Watch";
  return "Reallocate";
}

function pathFor(points, xFor, yFor) {
  return points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${xFor(point.week).toFixed(1)} ${yFor(point.value).toFixed(1)}`)
    .join(" ");
}

function matchAnalogs(catalog, draft) {
  return [...catalog]
    .map((item) => {
      const score =
        (item.category === draft.category ? 4 : 0) +
        (item.color === draft.color ? 4 : 0) +
        (item.fabric === draft.fabric ? 2 : 0);
      return { ...item, matchScore: score };
    })
    .filter((item) => item.matchScore > 0)
    .sort((a, b) => b.matchScore - a.matchScore || lifecycleTotal(b) - lifecycleTotal(a));
}

function credibleNewProductAnalogs(analogs) {
  return analogs.filter((item) => (item.matchScore || 0) * 10 >= 75);
}

function newProductCategoryBenchmark(catalog, draft) {
  const categoryTotals = catalog
    .filter((item) => item.category === draft.category)
    .map((item) => launchTotal(item))
    .filter((value) => Number.isFinite(value));

  return {
    median: Math.round(quantile(categoryTotals, 0.5)),
    topQuartile: Math.round(quantile(categoryTotals, 0.75)),
    count: categoryTotals.length,
  };
}

function newProductKey(draft) {
  return `${draft.category}||${draft.color}||${draft.fabric}`;
}

function predictNewProductDemand(draft, newProductModelArtifact, horizon) {
  const prediction = newProductModelArtifact?.predictionsByKey?.[newProductKey(draft)]?.[`future${horizon}wDemand`];
  return Number.isFinite(prediction) ? prediction : null;
}

function newProductModelBenchmark(draft, newProductModelArtifact) {
  return newProductModelArtifact?.categoryLaunchBenchmarks?.[draft.category] || null;
}

function decideNewProductAction({ forecastTotal, confidenceScore, credibleCount, benchmark }) {
  const weakAnalogSupport = credibleCount < MIN_CREDIBLE_ANALOGS;
  const lowConfidence = confidenceScore < 55;
  const lowDemand = forecastTotal < Math.round(benchmark.median * 0.75);
  const strongDemand = forecastTotal >= benchmark.topQuartile;
  const highConfidence = confidenceScore >= 70;

  if (weakAnalogSupport || lowConfidence || lowDemand) return "Watch";
  if (strongDemand && highConfidence) return "Restock";
  return "Reallocate";
}

function matchHistoricalAnalogs(catalog, product, limit = 8) {
  return [...catalog]
    .filter((item) => item.id !== product.id)
    .map((item) => ({ ...item, matchScore: analogMatchScore(product, item) }))
    .filter((item) => item.matchScore >= 75)
    .sort((a, b) => b.matchScore - a.matchScore || launchTotal(b) - launchTotal(a))
    .slice(0, limit);
}

function forecastFromAnalogs(analogs) {
  if (analogs.length < MIN_CREDIBLE_ANALOGS) {
    return { values: [], labels: [], window: "No forecast" };
  }

  const maxWeek = Math.min(MAX_LIFECYCLE_WEEK, Math.max(...analogs.map((item) => lifecycleSales(item).length - 1)));
  const labels = Array.from({ length: maxWeek + 1 }, (_, week) => week)
    .filter((week) => analogs.filter((item) => Number.isFinite(lifecycleSales(item)[week])).length >= MIN_CREDIBLE_ANALOGS);
  const lastWeek = labels.at(-1);
  const forecastLabels = Number.isFinite(lastWeek) ? Array.from({ length: lastWeek + 1 }, (_, week) => week) : [];
  const values = forecastLabels.map((week) => {
    const weekValues = analogs
      .map((item) => lifecycleSales(item)[week])
      .filter((value) => Number.isFinite(value));
    return Math.round(weekValues.reduce((sum, value) => sum + value, 0) / weekValues.length);
  });

  return {
    values,
    labels: forecastLabels,
    window: forecastLabels.length ? `W0-W${forecastLabels.at(-1)}` : "No forecast",
  };
}

function quantile(values, q) {
  const sorted = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (!sorted.length) return 0;
  const position = (sorted.length - 1) * q;
  const base = Math.floor(position);
  const rest = position - base;
  return sorted[base + 1] === undefined ? sorted[base] : sorted[base] + rest * (sorted[base + 1] - sorted[base]);
}

function TrendChart({
  title,
  subtitle,
  labels,
  actualPoints,
  forecastPoints,
  bandPoints,
  highlightWeeks,
  weeklyTiles,
  tileMode = "auto",
  totalLabel = "Forecast units",
  totalValue,
  comparisonPoints = [],
  comparisonLabel = "Current product actual sales",
  forecastTooltip = "Projected unit sales for the displayed forecast window, calculated from the active forecast method for this view.",
  analogRangeTooltip = "Historical analog range. For each week, the shaded band runs from the 25th percentile to the 75th percentile of matched analog products' weekly sales.",
}) {
  const width = 1040;
  const height = 320;
  const pad = 42;
  const leftPad = 54;
  const rightPad = 22;
  const values = [...actualPoints, ...forecastPoints, ...comparisonPoints, ...bandPoints.flat()].map((point) => point.value);
  const max = Math.max(...values, 1);
  const axisMax = Math.ceil(max / 100) * 100;
  const yTicks = [axisMax, Math.round(axisMax * 0.67), Math.round(axisMax * 0.33), 0];
  const minWeek = Math.min(...labels);
  const maxWeek = Math.max(...labels);
  const xFor = (week) => leftPad + ((week - minWeek) * (width - leftPad - rightPad)) / Math.max(1, maxWeek - minWeek);
  const yFor = (value) => height - pad - (value / axisMax) * (height - pad * 2);
  const topBand = bandPoints[0] || [];
  const bottomBand = bandPoints[1] || [];
  const bandPath = topBand.length
    ? `${pathFor(topBand, xFor, yFor)} L ${bottomBand.map((p) => `${xFor(p.week).toFixed(1)} ${yFor(p.value).toFixed(1)}`).reverse().join(" L ")} Z`
    : "";
  const start = highlightWeeks.length ? xFor(highlightWeeks[0]) : 0;
  const end = highlightWeeks.length ? xFor(highlightWeeks.at(-1)) : 0;
  const displayedTotal = totalValue ?? sumWeeks(Object.fromEntries(forecastPoints.map((p) => [p.week, p.value])), highlightWeeks);
  const tileColumnCount = Math.max(1, maxWeek - minWeek + 1);

  return (
    <div className="panel chart-panel">
      <div className="panel-heading">
        <div>
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
        <div className="chart-total">
          <span>{totalLabel}</span>
          <strong>{formatK(displayedTotal)}</strong>
        </div>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="trend-chart" role="img">
        {yTicks.map((tick) => (
          <g key={tick}>
            <line x1={leftPad} x2={width - rightPad} y1={yFor(tick)} y2={yFor(tick)} />
            <text x={leftPad - 10} y={yFor(tick) + 4} className="y-axis-label">{formatK(tick)}</text>
          </g>
        ))}
        {bandPath && <path d={bandPath} className="confidence-band" />}
        {!!highlightWeeks.length && <rect x={start} y={pad} width={Math.max(8, end - start)} height={height - pad * 2} className="horizon-window" />}
        {!!actualPoints.length && <path d={pathFor(actualPoints, xFor, yFor)} className="actual-line" />}
        {!!comparisonPoints.length && <path d={pathFor(comparisonPoints, xFor, yFor)} className="comparison-line" />}
        {!!forecastPoints.length && <path d={pathFor(forecastPoints, xFor, yFor)} className="forecast-line" />}
        {labels.map((week) => {
          const actual = actualPoints.find((point) => point.week === week);
          const forecast = forecastPoints.find((point) => point.week === week);
          const point = actual || forecast;
          return (
            <g key={week}>
              {point && <circle cx={xFor(week)} cy={yFor(point.value)} r={actual ? 4.5 : 3.2} className={actual ? "actual-dot" : "forecast-dot"} />}
              {point && (
                <text
                  x={xFor(week)}
                  y={Math.max(14, yFor(point.value) - 10)}
                  className={`point-value ${actual ? "actual-value" : "forecast-value"}`}
                >
                  {formatK(point.value)}
                </text>
              )}
              <text x={xFor(week)} y={height - 3}>W{week}</text>
            </g>
          );
        })}
      </svg>
      <div className="legend">
        {!!actualPoints.length && <span><i className="actual-key" />Actual sales</span>}
        {!!comparisonPoints.length && <span><i className="comparison-key" />{comparisonLabel}</span>}
        {!!forecastPoints.length && <span><i className="forecast-key" /><TooltipLabel label="Forecast" body={forecastTooltip} /></span>}
        {!!bandPath && <span><i className="band-key" /><TooltipLabel label="Analog range" body={analogRangeTooltip} /></span>}
      </div>
      <div className={`weekly-actuals ${tileMode === "twelve" ? "twelve-tiles" : ""}`} style={{ "--tile-columns": tileColumnCount }}>
        {weeklyTiles.map((tile) => (
          <div
            key={tile.label}
            className={tile.kind}
            style={Number.isFinite(tile.week) ? { gridColumn: `${tile.week - minWeek + 1} / span 1` } : undefined}
          >
            <span>{tile.label}</span>
            <strong>{formatK(tile.value)}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function RecommendationPanel({ action, confidence, lead, metrics, discountRisk, detailItems, rationaleText }) {
  const [isOpen, setIsOpen] = useState(false);
  const Icon = actionIcon[action] || PackageCheck;
  const actionClass = action.toLowerCase().replace(" ", "-");
  const opportunity = action === "Restock"
    ? "High demand opportunity"
    : action === "Watch"
      ? "Review before buying"
      : action === "Markdown Review"
        ? "Margin risk review"
        : "Allocation opportunity";
  const fallbackDecisionCopy = {
    Restock: "Choose Restock when forecasted demand clears the relevant top-quartile benchmark, confidence is high enough, and discount pressure is not the main reason for sales.",
    Watch: "Choose Watch when confidence is weak, forecasted demand is small versus the relevant benchmark, analog support is thin, or discount pressure may be inflating sales.",
    Reallocate: "Choose Reallocate when demand exists, but the safer first action is shifting inventory across stores before placing a larger order.",
    "Markdown Review": "Choose Markdown Review when average discount is 20% or higher and sales volume may be margin-driven rather than organic demand.",
  };

  return (
    <>
      <aside className="panel recommendation">
        <div className={`recommendation-callout ${actionClass}`}>
          <div className="callout-icon">
            {action === "Restock" ? <ShoppingCart size={22} /> : <Icon size={22} />}
          </div>
          <div>
            <strong>{action.toUpperCase()}</strong>
            <span>{opportunity}</span>
          </div>
        </div>
        <div className="recommendation-details">
          {metrics.map((metric) => (
            <div key={metric.label}>
              <span>{metric.label}</span>
              <strong>{metric.value}</strong>
            </div>
          ))}
        </div>
        <p className="rec-lead">{lead}</p>
        <div className={`warning ${discountRisk ? "show" : ""}`}>
          <AlertTriangle size={16} />
          {discountRisk
            ? "Discount may inflate observed demand. Treat sales as medium-confidence."
            : "Discount pressure is limited; demand signal is cleaner."}
        </div>
        <button className="recommendation-link" type="button" onClick={() => setIsOpen(true)}>
          View recommendation details <ChevronRight size={15} />
        </button>
      </aside>
      {isOpen && (
        <div className="modal-backdrop" role="presentation" onClick={() => setIsOpen(false)}>
          <section className="recommendation-modal" role="dialog" aria-modal="true" aria-labelledby="recommendation-modal-title" onClick={(event) => event.stopPropagation()}>
            <div className="modal-heading">
              <div>
                <span>Recommendation rationale</span>
                <h2 id="recommendation-modal-title">Why {action}?</h2>
              </div>
              <button type="button" aria-label="Close recommendation details" onClick={() => setIsOpen(false)}><X size={18} /></button>
            </div>
            <div className={`modal-action ${actionClass}`}>
              <Icon size={18} />
              <strong>{rationaleText || fallbackDecisionCopy[action] || fallbackDecisionCopy.Watch}</strong>
            </div>
            <div className="modal-evidence">
              {detailItems.map((item) => (
                <div key={item.label}>
                  <span><TooltipLabel label={item.label} body={item.logic} /></span>
                  <strong>{item.value}</strong>
                  <p>{item.note}</p>
                </div>
              ))}
            </div>
            <div className="modal-guardrail">
              <AlertTriangle size={16} />
              <p>This is a planner support signal, not an automatic buying decision. Exact order quantity and price still need business constraints.</p>
            </div>
          </section>
        </div>
      )}
    </>
  );
}

function DemandOutlookPanel({
  forecastTotal,
  forecastDisplay,
  forecastLabel = "Forecasted units",
  categoryMedian,
  restockEvents,
  confidenceScore,
  delta = 0,
  intro = "Next 12-week demand signal based on actual sales and matched analogs.",
  forecastWindow = "Next 12 weeks",
  forecastTooltip = "Projected units for the displayed forecast window, calculated from the active forecast method for this view.",
  benchmarkTooltip = "Median W0-W11 units sold for products in the same category. It is used as a baseline to judge whether this item is above or below typical category demand.",
  restockLabel = "Restock signal",
  restockValue,
  restockSubLabel,
  restockTooltip = "Count of historical restock events available in the Visuelle data for this product. It is treated as a directional replenishment signal, not current stock on hand.",
  insight,
  modelOutputs,
  confidenceTooltip,
}) {
  const confidence = confidenceLabel(confidenceScore);
  const tooltips = {
    forecast: forecastTooltip,
    benchmark: benchmarkTooltip,
    restock: restockTooltip,
    confidence: confidenceTooltip || "Evidence confidence score, not a guarantee for this exact product. Existing ML forecasts use held-out model test accuracy as the confidence signal. New products score credible analog count and analog consistency.",
  };
  const resolvedInsight = insight ?? (
    delta >= 0
      ? `Forecast is ${delta}% above the observed W0-W11 launch result.`
      : `Forecast is ${Math.abs(delta)}% below the observed W0-W11 launch result.`
  );

  return (
    <div className="controls panel outlook-panel demand-outlook">
      <div className="outlook-titleless">
        <p>{intro}</p>
        <LineChart size={20} />
      </div>
      <div className="outlook-metrics">
        <div>
          <span><TooltipLabel label={forecastLabel} body={tooltips.forecast} /></span>
          <strong>{forecastDisplay ?? formatK(forecastTotal)}</strong>
          <small>{forecastWindow}</small>
        </div>
        <div>
          <span><TooltipLabel label="Category benchmark" body={tooltips.benchmark} /></span>
          <strong>{formatK(categoryMedian)}</strong>
          <small>median</small>
        </div>
        <div>
          <span><TooltipLabel label={restockLabel} body={tooltips.restock} /></span>
          <strong>{restockValue ?? (restockEvents ? formatK(restockEvents) : "None")}</strong>
          <small>{restockSubLabel ?? (restockEvents ? "events" : "no history")}</small>
        </div>
        <div className="confidence-meter">
          <span><TooltipLabel label="Confidence" body={tooltips.confidence} /></span>
          <div className="confidence-ring" style={{ "--score": confidenceScore }}>
            <strong>{confidenceScore}%</strong>
          </div>
          <small>{confidence}</small>
        </div>
      </div>
      <div className="outlook-insight">
        <TrendingUp size={15} />
        <span>{resolvedInsight}</span>
      </div>
      {!!modelOutputs?.length && (
        <div className="model-output-strip">
          {modelOutputs.map((item) => (
            <div key={item.label}>
              <span>{item.label}</span>
              <strong>{item.value}</strong>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function TooltipLabel({ label, body }) {
  return (
    <span className="tooltip-label">
      {label}
      <button type="button" aria-label={`${label} explanation`}>
        <HelpCircle size={12} />
      </button>
      <span className="tooltip-bubble" role="tooltip">{body}</span>
    </span>
  );
}

function ProductSelector({ products, selectedId, setSelectedId }) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <img className="brand-logo" src="/visuelle-icon.png" alt="" />
        <div><strong>Visuelle</strong><span>Demand Cockpit</span></div>
      </div>
      <label className="search"><Search size={16} /><input value="Historical products" readOnly /></label>
      <div className="product-list">
        {products.map((product) => (
          <button key={product.id} className={`product-row ${selectedId === product.id ? "selected" : ""}`} onClick={() => setSelectedId(product.id)}>
            <img src={product.image} alt={`Product ${product.id}`} />
            <div><strong>{product.id}</strong><span>{product.category}</span><small>{product.color} / {product.fabric}</small></div>
            <ChevronRight size={16} />
          </button>
        ))}
      </div>
      <div className="sidebar-footer">
        <button>View all products</button>
        <nav>
          <span className="active">Cockpit</span>
          <span>Analytics</span>
          <span>Catalog</span>
          <span>Stores</span>
          <span>Reports</span>
        </nav>
      </div>
    </aside>
  );
}

function AnalogCards({ analogs, product, compactScroll = false, enablePopup = false }) {
  const [selectedAnalog, setSelectedAnalog] = useState(null);
  const canOpenAnalog = enablePopup || !!lifecycleSales(product).length;

  const openAnalog = (event, item) => {
    if (!canOpenAnalog) return;
    if (event.target.closest(".tooltip-label")) return;
    setSelectedAnalog(item);
  };

  return (
    <>
      <section className={`panel analog-panel ${compactScroll ? "compact-scroll" : ""}`}>
        <div className="panel-heading">
          <div><h2>Historical analogs</h2><p>One-row view of catalog products with Match score of at least 75. Click a product to inspect actual sales.</p></div>
        </div>
        <div className={`analog-grid ${compactScroll ? "scroll-row" : ""}`}>
          {analogs.map((item) => {
            const score = analogMatchScore(product, item);
            const matchScoreTooltip = product
              ? "Similarity to the selected product. Formula: same category = 40 points, same color = 20, same fabric = 25, same season = 15. Maximum score is 100%."
              : "Similarity to the selected new product attributes. Formula: same category = 40 points, same color = 40, same fabric = 20. Season is not used for new-product matching. Maximum score is 100%.";
            return (
              <article
                key={item.id}
                className={`analog-card ${canOpenAnalog ? "clickable" : ""}`}
                role={canOpenAnalog ? "button" : undefined}
                tabIndex={canOpenAnalog ? 0 : undefined}
                aria-label={canOpenAnalog ? `Open actual historical sales for analog product ${item.id}` : undefined}
                onClick={(event) => openAnalog(event, item)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    openAnalog(event, item);
                  }
                }}
              >
                <img src={item.image} alt={`Analog product ${item.id}`} />
                <div><strong>{item.id}</strong><span>{item.category}</span><small>{item.color} / {item.fabric}</small></div>
                <div className="analog-metrics"><span>Actual lifecycle</span><strong>{formatK(lifecycleTotal(item))}</strong></div>
                <div className="match-score">
                  <span><TooltipLabel label="Match score" body={matchScoreTooltip} /></span>
                  <strong>{score}%</strong>
                </div>
              </article>
            );
          })}
        </div>
      </section>
      {canOpenAnalog && selectedAnalog && (
        <AnalogSalesModal analog={selectedAnalog} product={product} analogs={analogs} onClose={() => setSelectedAnalog(null)} />
      )}
    </>
  );
}

function AnalogSalesModal({ analog, product, analogs, onClose }) {
  const analogLifecycle = lifecycleSales(analog);
  const productLifecycle = product ? lifecycleSales(product) : [];
  const maxWeek = Math.max(
    analogLifecycle.length - 1,
    productLifecycle.length - 1,
    ...analogs.map((item) => lifecycleSales(item).length - 1),
    0
  );
  const labels = Array.from({ length: maxWeek + 1 }, (_, week) => week);
  const actualPoints = lifecyclePoints(analogLifecycle);
  const comparisonPoints = productLifecycle.length ? lifecyclePoints(productLifecycle) : [];
  const [analogTop, analogBottom] = analogRangeByWeek(analogs, labels);
  const total = analogLifecycle.reduce((sum, value) => sum + value, 0);
  const peakValue = Math.max(...analogLifecycle, 0);
  const peakWeek = analogLifecycle.indexOf(peakValue);
  const matchScore = product ? analogMatchScore(product, analog) : Math.min(100, Math.round((analog.matchScore || 0) * 10));
  const actualWindow = lifecycleWindow(analogLifecycle);

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <section className="analog-modal" role="dialog" aria-modal="true" aria-labelledby="analog-modal-title" onClick={(event) => event.stopPropagation()}>
        <div className="modal-heading">
          <div>
            <span>Actual historical sales</span>
            <h2 id="analog-modal-title">Product {analog.id}</h2>
          </div>
          <button type="button" aria-label="Close analog sales" onClick={onClose}><X size={18} /></button>
        </div>
        <div className="analog-modal-summary">
          <img src={analog.image} alt={`Analog product ${analog.id}`} />
          <div>
            <p>{analog.category} <i /> {analog.color} <i /> {analog.fabric} <i /> {analog.season}</p>
            <dl>
              <div><dt>Total {actualWindow}</dt><dd>{formatK(total)}</dd></div>
              <div><dt>Peak week</dt><dd>W{peakWeek}</dd></div>
              <div><dt>Peak units</dt><dd>{formatK(peakValue)}</dd></div>
              <div><dt>Match score</dt><dd>{matchScore}%</dd></div>
            </dl>
          </div>
        </div>
        <TrendChart
          title="Actual historical sales"
          subtitle={product ? `Observed ${actualWindow} unit sales for analog product ${analog.id}, compared with current product ${product.id} and the matched analog range.` : `Observed ${actualWindow} unit sales for analog product ${analog.id}, compared with the matched analog range.`}
          labels={labels}
          actualPoints={actualPoints}
          forecastPoints={[]}
          bandPoints={[analogTop, analogBottom]}
          highlightWeeks={[]}
          comparisonPoints={comparisonPoints}
          comparisonLabel={product ? `Current product ${product.id}` : "Current product"}
          analogRangeTooltip="Analog range through the available lifecycle window, capped at W52. For each week, the shaded band is the 25th to 75th percentile of matched analog products with data for that week."
          weeklyTiles={analogLifecycle.map((value, week) => ({ label: `W${week}`, value, kind: "observed" }))}
          totalLabel="Total units"
          totalValue={total}
        />
      </section>
    </div>
  );
}

function StoreSignals({ stores, forecastTotal, forecastWindow = "future 12 weeks", productAction = "Reallocate" }) {
  const [isOpen, setIsOpen] = useState(false);
  const totalAllocationWeight = stores.reduce((sum, store) => sum + (store.allocationWeight || 0), 0);
  const actualToDateTotal = stores.reduce((sum, store) => sum + (store.actualToDate || store.total12 || 0), 0);
  const hasForecast = Number.isFinite(forecastTotal);
  const canAllocate = productAction === "Restock" || productAction === "Reallocate";
  const allocationSignal = productAction === "Restock" ? "Restock candidate" : "Allocation candidate";
  const conservativeSignal = productAction === "Markdown Review" ? "Hold / margin review" : "Monitor only";
  const signalTooltip = hasForecast
    ? `Directional allocation forecast for all stores with observed sales. Formula: product ${forecastWindow} forecast (${formatK(forecastTotal)} units) x store allocation weight. Allocation weight = 70% actual-to-date store share + 30% recent-4-week store share, capped at 2.5x actual-to-date share and normalized across stores. Store labels are gated by the product-level recommendation: Restock can show Restock candidate, Reallocate can show Allocation candidate, and Watch shows Monitor only. Inventory depth and store profiles are not available, so this is not an exact store order quantity.`
    : `Directional store signal from actual-to-date store sales only. Formula: each store's actual units as a share of all observed store units (${formatK(actualToDateTotal)} units). Inventory depth and store profiles are not available, so this is not an exact store order quantity.`;
  const baseRows = stores.map((store) => {
    const fallbackWeight = actualToDateTotal ? (store.actualToDate || store.total12 || 0) / actualToDateTotal : 0;
    const allocationWeight = totalAllocationWeight ? (store.allocationWeight || 0) / totalAllocationWeight : fallbackWeight;
    return {
      ...store,
      allocationWeight,
      storeForecast: hasForecast ? Math.round(forecastTotal * allocationWeight) : null,
      storeShare: `${Math.round(allocationWeight * 100)}%`,
    };
  });
  const forecastValues = baseRows.map((store) => store.storeForecast).filter((value) => Number.isFinite(value));
  const topThreshold = quantile(forecastValues, 0.75);
  const lowThreshold = quantile(forecastValues, 0.25);
  const rows = baseRows
    .map((store) => ({
      ...store,
      signal: !canAllocate
        ? (hasForecast && store.storeForecast <= lowThreshold && !store.recent4 ? "Hold / transfer review" : conservativeSignal)
        : hasForecast && store.storeForecast >= topThreshold
          ? allocationSignal
          : hasForecast && store.storeForecast <= lowThreshold && !store.recent4
            ? "Watch / possible transfer"
          : "Monitor",
    }))
    .sort((a, b) => (b.storeForecast || 0) - (a.storeForecast || 0) || (b.actualToDate || 0) - (a.actualToDate || 0));

  return (
    <>
      <section className="panel store-panel">
        <div className="panel-heading"><div><h2>Store-level allocation signals</h2><p>Directional signals because inventory depth and store profiles are missing.</p></div><Store size={20} /></div>
        <div className="store-table">
          <button className="view-stores-button" type="button" onClick={() => setIsOpen(true)}>View all {rows.length} stores</button>
          <div className="store-head">
            <span>Store</span>
            <span>Actual</span>
            <span>Recent</span>
            <span><TooltipLabel label={hasForecast ? `Forecast ${forecastWindow}` : "Actual share"} body={signalTooltip} /></span>
            <span>Signal</span>
          </div>
          {rows.slice(0, 5).map((store) => (
            <div className="store-row" key={store.store}>
              <strong>{store.store}</strong><span>{store.actualToDate || store.total12}</span><span>{store.recent4 ?? 0}</span><span>{hasForecast ? store.storeForecast : store.storeShare}</span><em className={classForSignal(store.signal)}>{store.signal}</em>
            </div>
          ))}
        </div>
      </section>
      {isOpen && (
        <div className="modal-backdrop" role="presentation" onClick={() => setIsOpen(false)}>
          <section className="store-modal" role="dialog" aria-modal="true" aria-labelledby="store-modal-title" onClick={(event) => event.stopPropagation()}>
            <div className="modal-heading">
              <div>
                <span>Store signals</span>
                <h2 id="store-modal-title">All {rows.length} stores</h2>
              </div>
              <button type="button" aria-label="Close store details" onClick={() => setIsOpen(false)}><X size={18} /></button>
            </div>
            <p className="store-modal-note">{signalTooltip}</p>
            <div className="store-table modal-store-table">
              <div className="store-head"><span>Store</span><span>Actual to date</span><span>Recent 4W</span><span>{hasForecast ? `Forecast ${forecastWindow}` : "Actual share"}</span><span>Signal</span></div>
              {rows.map((store) => (
                <div className="store-row" key={store.store}>
                  <strong>{store.store}</strong><span>{store.actualToDate || store.total12}</span><span>{store.recent4 ?? 0}</span><span>{hasForecast ? store.storeForecast : store.storeShare}</span><em className={classForSignal(store.signal)}>{store.signal}</em>
                </div>
              ))}
            </div>
          </section>
        </div>
      )}
    </>
  );
}

function Benchmark({ label, values, avgDiscount }) {
  const max = Math.max(...values.map((item) => item.value), 1);
  const discountText = Number.isFinite(avgDiscount)
    ? `Avg. discount is ${Math.round(avgDiscount * 100)}%. Review if it approaches the 20% risk threshold before converting demand signal into a buying decision.`
    : "Discount data is not available for this new product simulation. Review margin assumptions before converting demand signal into a buying decision.";
  return (
    <section className="panel benchmark">
      <div className="panel-heading"><div><h2>Category benchmark</h2><p>{label}</p></div><TrendingUp size={20} /></div>
      {values.map((item) => (
        <div className="bar-row" key={item.label}>
          <span>{item.label}</span><div><i className={item.kind} style={{ width: `${Math.max(4, (item.value / max) * 100)}%` }} /></div><strong>{formatK(item.value)}</strong>
        </div>
      ))}
      <div className="markdown-impact">
        <AlertTriangle size={16} />
        <div>
          <strong>Discount & margin watch</strong>
          <p>{discountText}</p>
        </div>
      </div>
    </section>
  );
}

function AttributeSelect({ label, value, options, onChange }) {
  return (
    <label className="attribute-select">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => <option key={option} value={option}>{option}</option>)}
      </select>
    </label>
  );
}

function ExistingProductView({ data, product, existingModel }) {
  const actualLifecycle = lifecycleSales(product);
  const actualPoints = lifecyclePoints(actualLifecycle);
  const actualLifecycleTotal = actualLifecycle.reduce((sum, value) => sum + value, 0);
  const actualWindow = lifecycleWindow(actualLifecycle);
  const displayAnalogs = matchHistoricalAnalogs(data.newProductModel.catalog, product, 8);
  const forecast = calculateExistingProductForecast(product, displayAnalogs, data.newProductModel.catalog, existingModel);
  const labels = [...actualLifecycle.map((_, i) => i), ...forecast.points.map((point) => point.week)];
  const evidenceProduct = { ...product, analogs: displayAnalogs };
  const [actualAnalogTop, actualAnalogBottom] = analogRangeByWeek(displayAnalogs, actualLifecycle.map((_, i) => i));
  const analogTop = [...actualAnalogTop, ...forecast.top];
  const analogBottom = [...actualAnalogBottom, ...forecast.bottom];
  const confidenceScore = forecast.confidenceScore || calculateEvidenceConfidence(evidenceProduct);
  const categoryBenchmark = categoryBenchmarks(data, product);
  const recommendedAction = decideExistingProductAction({ forecast, avgDiscount: product.avgDiscount, categoryBenchmark });
  const trendMultiplier = `${forecast.trendAdjustment.toFixed(2)}x`;
  const trendDirection = forecast.trendAdjustment > 1.01 ? "uplift" : forecast.trendAdjustment < 0.99 ? "downshift" : "neutral";
  const trendKeys = forecast.trendSignal?.keys
    ? `${forecast.trendSignal.keys.category}, ${forecast.trendSignal.keys.color}, ${forecast.trendSignal.keys.fabric}`
    : "category, color, fabric";
  const next4wDemand = forecast.modelPredictions?.next4w;
  const next8wDemand = forecast.modelPredictions?.next8w;
  const next12wDemand = forecast.modelPredictions?.next12w ?? forecast.total;
  const modelCategoryBenchmark = forecast.modelPredictions?.categoryBenchmarkNext12;
  const actualEndWeek = actualLifecycle.length ? actualLifecycle.length - 1 : 0;
  const isModelForecast = forecast.source === "ML model";
  const forecastMilestonePoints = isModelForecast
    ? [
      { week: actualEndWeek + 4, value: Math.round(next4wDemand || 0), total: Math.round(next4wDemand || 0), label: "Next 4w" },
      { week: actualEndWeek + 8, value: Math.round(next8wDemand || 0), total: Math.round(next8wDemand || 0), label: "Next 8w" },
      { week: actualEndWeek + 12, value: Math.round(next12wDemand || 0), total: Math.round(next12wDemand || 0), label: "Next 12w" },
    ].filter((point) => Number.isFinite(point.value))
    : forecast.points;
  const chartForecastPoints = isModelForecast
    ? [{ week: actualEndWeek, value: actualLifecycle[actualEndWeek] || 0 }, ...forecastMilestonePoints]
    : forecastMilestonePoints;
  const chartHighlightWeeks = isModelForecast
    ? [actualEndWeek, ...forecastMilestonePoints.map((point) => point.week)]
    : forecastMilestonePoints.map((point) => point.week);
  const chartLabels = isModelForecast
    ? [...actualLifecycle.map((_, i) => i), ...forecastMilestonePoints.map((point) => point.week)]
    : labels;
  const chartBandPoints = isModelForecast ? [actualAnalogTop, actualAnalogBottom] : [analogTop, analogBottom];
  const chartWeeklyTiles = isModelForecast
    ? [
      ...actualLifecycle.map((value, week) => ({ label: `W${week}`, week, value, kind: "observed" })),
      ...forecastMilestonePoints.map((point) => ({ label: `${point.label} total`, week: point.week, value: point.total, kind: "forecast-tile" })),
    ]
    : [
      ...actualLifecycle.map((value, week) => ({ label: `W${week}`, week, value, kind: "observed" })),
      ...forecast.points.map((point) => ({ label: `W${point.week}`, week: point.week, value: point.value, kind: "forecast-tile" })),
    ];
  const confidenceBreakdown = forecast.confidenceBreakdown || {};
  const confidenceBreakdownText = forecast.source === "ML model"
    ? `model test accuracy ${forecast.modelPredictions?.testAccuracyPct ?? "N/A"}%, WAPE ${forecast.modelPredictions?.testWapePct ?? "N/A"}%, bias ${forecast.modelPredictions?.testBiasPct ?? "N/A"}%`
    : `actual history ${confidenceBreakdown.actualHistory ?? 0}/10, matched analogs ${confidenceBreakdown.matchedAnalogs ?? 0}/10, matched tail ${confidenceBreakdown.matchedTail ?? 0}/35, category fallback ${confidenceBreakdown.categoryFallback ?? 0}/10, consistency ${confidenceBreakdown.consistency ?? 0}/15, Google Trends coverage ${confidenceBreakdown.trendCoverage ?? 0}/5`;
  const decisionMedian = isModelForecast ? (modelCategoryBenchmark || categoryBenchmark.median) : forecast.candidateMedian;
  const decisionTopQuartile = isModelForecast ? (modelCategoryBenchmark ? modelCategoryBenchmark * 1.25 : categoryBenchmark.topQuartile) : forecast.candidateTopQuartile;
  const lowDemandThreshold = Math.round(decisionMedian * 0.75);
  const hasStrongForecast = forecast.total >= decisionTopQuartile;
  const hasEnoughCoverage = forecast.matchedCoverageWeeks >= 6;
  const hasHighConfidence = confidenceScore >= 70;
  const watchReason = confidenceScore < 50
    ? `confidence is ${confidenceScore}% (Watch if <50%)`
    : forecast.total < lowDemandThreshold
      ? `forecasted demand is ${formatK(forecast.total)} units, below the Watch threshold of ${formatK(lowDemandThreshold)}`
      : isModelForecast ? `the model signal does not pass the Restock threshold` : `the signal does not pass the Restock gates`;
  const modelDetailItems = [
    { label: "Model input", value: `Product ${product.id}`, note: "The planner selects a product ID; the system retrieves all related product information for prediction.", logic: "The user-facing input is product_id. Internal features are looked up from product data: attributes, observed sales momentum, discount/price, store coverage, restock signal, category benchmark, and Google Trends." },
    { label: "Model outputs", value: `${formatK(next12wDemand)} next 12w`, note: `Next 4w ${formatK(next4wDemand || 0)}, next 8w ${formatK(next8wDemand || 0)}, next 12w ${formatK(next12wDemand || 0)}.`, logic: "Three trained XGBoost regressors predict cumulative future demand for the next 4, 8, and 12 weeks. The chart markers and tiles show the same cumulative totals." },
    { label: "Model accuracy", value: `${forecast.modelPredictions?.testAccuracyPct ?? "N/A"}%`, note: `Held-out test WAPE is ${forecast.modelPredictions?.testWapePct ?? "N/A"}%; bias is ${forecast.modelPredictions?.testBiasPct ?? "N/A"}%.`, logic: "Accuracy is calculated as 1 - WAPE on the 15% product-level test set. The split is product-level 70/15/15, so test products are not seen during training." },
    { label: "Category benchmark", value: `${formatK(modelCategoryBenchmark || categoryBenchmark.median)} avg. next 12w`, note: `Recommendation compares next 12w model demand with the ${product.category} model category benchmark.`, logic: "The category benchmark is derived from the trained model artifact for the same next-12-week target, so it is on the same unit basis as the model forecast." },
    { label: "Watch threshold", value: `${formatK(lowDemandThreshold)} next 12w`, note: `Calculated as 75% of the ${product.category} model category benchmark (${formatK(modelCategoryBenchmark || categoryBenchmark.median)}).`, logic: "Watch threshold = category benchmark x 0.75. It marks a low-demand zone: if the next-12-week forecast is below this value, the product is watched instead of replenished because predicted demand is materially below the category baseline. It is not an order quantity." },
    { label: "Discount check", value: `${Math.round(product.avgDiscount * 100)}% avg. discount`, note: product.avgDiscount >= 0.2 ? "Discounting is material, so observed sales should not be treated as pure organic demand." : "Discount pressure is limited, so the observed sales signal is cleaner.", logic: "Discounts can inflate unit sales. Lower average discount means demand is more likely to reflect product appeal rather than markdown pressure." },
    { label: "Store/restock signal", value: `${product.stores} stores`, note: `${product.restockEvents || 0} restock events are included as model features and allocation context.`, logic: "Store coverage and restock history are internal model features for existing products. They support demand prediction but do not provide exact current stock on hand." },
  ];
  const analogDetailItems = [
    { label: "Actual lifecycle sales", value: `${formatK(actualLifecycleTotal)} units`, note: `Observed sales are available from ${actualWindow}; the forecast begins after the last observed week.`, logic: "Actual lifecycle sales are transaction-derived units by week since release. They anchor the scale of the forecast before any analog tail is used." },
    { label: "Forecasted demand", value: `${formatK(forecast.total)} units`, note: `${forecast.window} is generated from scaled future tails, then adjusted by Google Trends.`, logic: `Base forecast is ${formatK(forecast.baseTotal)} units from matched analog/category future week units x scale factor. Final forecast applies Google Trends ${trendMultiplier}. Scale factor blends total actual lifecycle scale and recent 4-week momentum, capped between 0.25x and 3.0x.` },
    { label: "Confidence score", value: `${confidenceScore}%`, note: `Directional evidence reliability: ${confidenceBreakdownText}.`, logic: "Confidence is not forecast accuracy. It is calculated from a 0-90 evidence score capped at 88%: base 10 points, up to 10 for actual-history coverage, up to 10 for matched analog count, up to 35 for future weeks with at least 3 matched analog tails, up to 10 for same-category fallback coverage, up to 15 for tail consistency, and 5 for sufficient Google Trends coverage." },
    { label: "Google Trends", value: trendMultiplier, note: `Weighted trend average moves from ${forecast.trendSignal?.recentAvg ?? "N/A"} to ${forecast.trendSignal?.forecastAvg ?? "N/A"} for ${trendKeys}.`, logic: "Trend adjustment compares category/color/fabric search interest in the forecast window against the previous 12 lifecycle weeks. Category weight is 50%, color is 25%, fabric is 25%. The adjustment is capped between 0.85x and 1.15x so search interest cannot override sales and analog evidence." },
    { label: "Tail coverage", value: `${forecast.matchedCoverageWeeks}/12 weeks`, note: "Weeks with at least 3 matched analog products are higher-quality evidence.", logic: "Matched analog tails use Match score >=75. If fewer than 3 matched analogs have data in a future week, the forecast falls back to same-category lifecycle tails and confidence is reduced." },
    { label: "Tail benchmark", value: `${formatK(forecast.candidateMedian)} median`, note: "Recommendation compares forecasted demand with the scaled analog/category tail distribution after the same Google Trends adjustment.", logic: "Tail benchmark is calculated from scaled historical future-tail totals for the same category, then multiplied by the same Google Trends adjustment as the forecast. Restock requires forecasted units to meet or exceed the adjusted top-quartile tail benchmark, confidence >=70%, and at least 6 of 12 weeks with 3+ matched analog tails. Watch is used when the forecast is below 75% of the adjusted median or confidence is weak." },
    { label: "Watch threshold", value: `${formatK(lowDemandThreshold)} units`, note: `Calculated as 75% of the adjusted tail median (${formatK(decisionMedian)}).`, logic: "Watch threshold = adjusted category/analog median x 0.75. It marks a low-demand zone: if the forecast is below this value, demand is materially weaker than the comparable historical tail and the planner should watch rather than replenish. It is not an order quantity." },
    { label: "Discount check", value: `${Math.round(product.avgDiscount * 100)}% avg. discount`, note: product.avgDiscount >= 0.2 ? "Discounting is material, so observed sales should not be treated as pure organic demand." : "Discount pressure is limited, so the observed sales signal is cleaner.", logic: "Discounts can inflate unit sales. Lower average discount means demand is more likely to reflect product appeal rather than markdown pressure." },
    { label: "Restock history", value: product.restockEvents ? `${product.restockEvents} restock events` : "No restock history", note: "Restock events support the replenishment context but do not prove current stock level.", logic: "Restock events are historical replenishment records in the Visuelle data. They are used as directional context only; exact inventory depth, current stock on hand, and store profiles are missing, so the app cannot recommend an exact order quantity." },
  ];
  const rationaleText = isModelForecast
    ? recommendedAction === "Restock"
      ? `Choose Restock because Product ID ${product.id} returns a next 12-week model forecast of ${formatK(forecast.total)} units, above the ${product.category} model category benchmark of ${formatK(modelCategoryBenchmark || categoryBenchmark.median)}, with ${confidenceScore}% model confidence based on held-out test accuracy.`
      : recommendedAction === "Watch"
        ? `Choose Watch because Product ID ${product.id} returns a next 12-week model forecast of ${formatK(forecast.total)} units, which is below the Watch threshold of ${formatK(lowDemandThreshold)} or model confidence is below the decision threshold.`
        : recommendedAction === "Markdown Review"
          ? `Choose Markdown Review because Product ID ${product.id} has a next 12-week model forecast of ${formatK(forecast.total)} units, but average discount is ${Math.round(product.avgDiscount * 100)}%, so demand may be margin-driven.`
          : `Choose Reallocate because Product ID ${product.id} returns a next 12-week model forecast of ${formatK(forecast.total)} units. It is above the Watch threshold but not strong enough versus the model category benchmark for an automatic Restock recommendation.`
    : recommendedAction === "Restock"
    ? `Choose Restock because the next 12-week forecast is ${formatK(forecast.total)} units (${forecast.window}), after a Google Trends ${trendMultiplier} ${trendDirection}, meets the adjusted top-quartile tail benchmark of ${formatK(forecast.candidateTopQuartile)}, has ${confidenceScore}% confidence (threshold >=70%), and has ${forecast.matchedCoverageWeeks}/12 weeks with 3+ matched analog tails (threshold >=6).`
    : recommendedAction === "Watch"
      ? `Choose Watch because the next 12-week forecast is ${formatK(forecast.total)} units (${forecast.window}) after a Google Trends ${trendMultiplier} ${trendDirection}, and ${watchReason}.`
      : recommendedAction === "Markdown Review"
        ? `Choose Markdown Review because the next 12-week forecast is ${formatK(forecast.total)} units (${forecast.window}) after a Google Trends ${trendMultiplier} ${trendDirection}, but average discount is ${Math.round(product.avgDiscount * 100)}%, so sales may be margin-driven rather than organic demand.`
        : hasStrongForecast && (!hasHighConfidence || !hasEnoughCoverage)
          ? `Choose Reallocate because the next 12-week forecast is ${formatK(forecast.total)} units (${forecast.window}) after a Google Trends ${trendMultiplier} ${trendDirection}, but it does not pass all Restock gates: confidence is ${confidenceScore}% (threshold >=70%) and matched tail coverage is ${forecast.matchedCoverageWeeks}/12 weeks (threshold >=6). Review allocation before placing a larger buy.`
          : `Choose Reallocate because the next 12-week forecast is ${formatK(forecast.total)} units (${forecast.window}) after a Google Trends ${trendMultiplier} ${trendDirection}; it is above the Watch threshold of ${formatK(lowDemandThreshold)} but below the Restock top-quartile threshold of ${formatK(forecast.candidateTopQuartile)}. Review store allocation before placing a larger buy.`;

  return (
    <>
      <header className="topbar">
        <div />
        <div className="summary-strip"><div><span>Dataset</span><strong>{data.summary.productCount.toLocaleString()} products</strong></div><div><span>Stores</span><strong>{data.summary.storeCount}</strong></div><div><span>Actuals</span><strong>{actualWindow}</strong></div></div>
      </header>
      <section className="decision-grid">
        <div className="product-card panel">
          <img src={product.image} alt={`Product ${product.id}`} />
          <div className="product-card-copy">
            <div className="product-title-row">
              <h2>Product {product.id}</h2>
              <span>{product.season}</span>
            </div>
            <p className="product-attributes">{product.category} <i /> {product.color} <i /> {product.fabric}</p>
            <p className="product-release"><span>Release: {formatDate(product.releaseDate)}</span><i /><span>Launch: W0</span></p>
            <dl>
              <div><dt>{actualWindow}</dt><dd>{formatK(actualLifecycleTotal)}</dd><small>actual units</small></div>
              <div><dt>Stores</dt><dd>{product.stores}</dd><small>selling</small></div>
              <div><dt>Avg. discount</dt><dd>{Math.round(product.avgDiscount * 100)}%</dd><small>observed</small></div>
            </dl>
          </div>
        </div>
        <DemandOutlookPanel
          forecastTotal={forecast.total}
          forecastLabel="Forecasted units"
          categoryMedian={categoryBenchmark.median}
          restockEvents={product.restockEvents}
          confidenceScore={confidenceScore}
          intro="Input is Product ID. The trained XGBoost model looks up product details and outputs next 4w, 8w, and 12w demand."
          forecastWindow={forecast.window}
          forecastTooltip={forecast.source === "ML model" ? `Product ID ${product.id} is used to retrieve product attributes, the full actual sales trend from W0 through the current week, lifecycle-stage features, discount/price, store coverage, restock signal, category benchmark, and Google Trends. The XGBoost model outputs next 4w, 8w, and 12w demand; the chart displays the 12w output. Test accuracy: ${forecast.modelPredictions?.testAccuracyPct ?? "N/A"}%.` : `Projected units for ${forecast.window}. Formula: scaled analog/category lifecycle forecast (${formatK(forecast.baseTotal)} units) x Google Trends adjustment (${trendMultiplier}). Trend uses weighted ${trendKeys} search interest in ${forecast.window} versus the previous 12 lifecycle weeks, capped between 0.85x and 1.15x.`}
          benchmarkTooltip={isModelForecast ? `Model category benchmark for ${product.category}, calculated from the trained next-12-week model artifact. It is on the same target basis as the model forecast.` : `Median W0-W11 launch units sold for ${product.category} products in the Visuelle catalog. This is category context only for existing products; the recommendation decision uses the adjusted future-tail benchmark shown in the rationale popup.`}
          insight={forecast.source === "ML model" ? `ML forecast is ${formatK(forecast.total)} units for ${forecast.window}; model backtest accuracy is ${forecast.modelPredictions?.testAccuracyPct ?? "N/A"}%.` : `Forecast is ${formatK(forecast.total)} units for ${forecast.window}; Google Trends applies a ${trendMultiplier} ${trendDirection}.`}
          modelOutputs={forecast.source === "ML model" ? [
            { label: "Next 4w", value: Number.isFinite(next4wDemand) ? formatK(next4wDemand) : "N/A" },
            { label: "Next 8w", value: Number.isFinite(next8wDemand) ? formatK(next8wDemand) : "N/A" },
            { label: "Next 12w", value: Number.isFinite(next12wDemand) ? formatK(next12wDemand) : "N/A" },
          ] : null}
        />
        <RecommendationPanel
          action={recommendedAction}
          confidence={confidenceLabel(confidenceScore)}
          lead={`Product ${product.id} has transaction-derived actual sales through ${actualWindow}; forecast starts at ${forecast.window.split("-")[0]}.`}
          metrics={[
            { label: "Recommended action", value: recommendedAction },
            { label: "Forecast window", value: forecast.window },
            { label: "Watch threshold", value: formatK(lowDemandThreshold) },
            { label: "Model accuracy", value: forecast.modelPredictions?.testAccuracyPct ? `${forecast.modelPredictions.testAccuracyPct}%` : "N/A" },
            { label: "Key driver", value: forecast.source === "ML model" ? "Lifecycle trend + operational signals" : forecast.matchedCoverageWeeks >= 6 ? "Matched analog tail demand" : "Category tail fallback" },
          ]}
          detailItems={isModelForecast ? modelDetailItems : analogDetailItems}
          rationaleText={rationaleText}
          discountRisk={product.avgDiscount >= 0.2}
        />
      </section>
      <section className="main-grid single-chart">
        <TrendChart
          title={isModelForecast ? "Actual lifecycle sales and 4w / 8w / 12w forecast" : "Actual lifecycle sales and 12-week forecast"}
          subtitle={isModelForecast ? `${actualWindow} are observed weekly sales. Forecast markers show cumulative model demand for next 4w, 8w, and 12w.` : `${actualWindow} are observed transaction sales. ${forecast.window} is forecast from scaled lifecycle tails with Google Trends ${trendMultiplier}.`}
          labels={chartLabels}
          actualPoints={actualPoints}
          forecastPoints={chartForecastPoints}
          bandPoints={chartBandPoints}
          highlightWeeks={chartHighlightWeeks}
          weeklyTiles={chartWeeklyTiles}
          totalLabel={isModelForecast ? "Next 12w total" : "Forecast units"}
          totalValue={forecast.total}
          forecastTooltip={isModelForecast ? `Forecast markers show cumulative model outputs: next 4w total, next 8w total, and next 12w total. These are the same totals shown in the tiles and Demand outlook.` : `Projected ${forecast.window} units. Each historical candidate tail is scaled to current product demand, then multiplied by Google Trends ${trendMultiplier}. Trend compares weighted ${trendKeys} search interest in the forecast window against the previous 12 lifecycle weeks.`}
          analogRangeTooltip={isModelForecast ? "Analog range is shown only over actual historical weeks as P25-P75 of matched analog actual lifecycle sales." : "Analog range covers actual and forecast windows. In actual weeks, it is the P25-P75 range of matched analog actual sales. In forecast weeks, it is the P25-P75 range of scaled historical future tails after the same Google Trends adjustment."}
        />
      </section>
      <section className="lower-grid"><StoreSignals stores={product.storeSignals} forecastTotal={next12wDemand} forecastWindow={isModelForecast ? "next 12w" : forecast.window} productAction={recommendedAction} /><Benchmark label={isModelForecast ? `${product.category} model category benchmark for the same next-12-week target.` : `${product.category} W0-W11 category benchmark across ${categoryBenchmark.count} category products. Product value uses available actual lifecycle sales.`} avgDiscount={product.avgDiscount} values={isModelForecast ? [{ label: "Model next 12w", value: next12wDemand, kind: "product" }, { label: "Category avg next 12w", value: modelCategoryBenchmark || categoryBenchmark.median, kind: "median" }, { label: "Restock threshold", value: (modelCategoryBenchmark || categoryBenchmark.median) * 1.25, kind: "top" }] : [{ label: "This product actual", value: actualLifecycleTotal, kind: "product" }, { label: "Category top quartile", value: categoryBenchmark.topQuartile, kind: "top" }, { label: "Category median", value: categoryBenchmark.median, kind: "median" }]} /><AnalogCards analogs={displayAnalogs} product={product} /></section>
    </>
  );
}

function NewProductView({ data, draft, setDraft, newProductModelArtifact }) {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [imageStatus, setImageStatus] = useState("idle");
  const [imageMessage, setImageMessage] = useState("");
  const analogs = useMemo(() => matchAnalogs(data.newProductModel.catalog, draft), [data, draft]);
  const credibleAnalogs = useMemo(() => credibleNewProductAnalogs(analogs), [analogs]);
  const modelPredictions = {
    future4w: predictNewProductDemand(draft, newProductModelArtifact, 4),
    future8w: predictNewProductDemand(draft, newProductModelArtifact, 8),
    future12w: predictNewProductDemand(draft, newProductModelArtifact, 12),
  };
  const forecastAvailable = [modelPredictions.future4w, modelPredictions.future8w, modelPredictions.future12w].every(Number.isFinite);
  const selectedTotal = forecastAvailable ? modelPredictions.future12w : 0;
  const forecastWindow = "Next 12w";
  const modelBenchmark = newProductModelBenchmark(draft, newProductModelArtifact);
  const benchmark = modelBenchmark || newProductCategoryBenchmark(data.newProductModel.catalog, draft);
  const validation = newProductModelArtifact?.validationSummary?.["12"];
  const modelConfidence = Number.isFinite(validation?.testAccuracyPct) ? Math.round(validation.testAccuracyPct) : null;
  const analogConfidence = calculateNewProductConfidence(credibleAnalogs);
  const watchThreshold = Math.round(benchmark.median * 0.75);
  const confidenceScore = forecastAvailable
    ? Math.max(25, Math.min(85, credibleAnalogs.length < MIN_CREDIBLE_ANALOGS ? Math.min(45, modelConfidence ?? analogConfidence) : modelConfidence ?? analogConfidence))
    : analogConfidence;
  const confidence = confidenceLabel(confidenceScore);
  const action = decideNewProductAction({
    forecastTotal: selectedTotal,
    confidenceScore,
    credibleCount: credibleAnalogs.length,
    benchmark,
  });
  const rationaleText = action === "Restock"
    ? `Choose Restock because the XGBoost model predicts ${formatK(selectedTotal)} units for the first 12 weeks, above the ${draft.category} W0-W11 top-quartile benchmark of ${formatK(benchmark.topQuartile)}, with ${credibleAnalogs.length} credible analogs (Match score >=75) and ${confidenceScore}% confidence (threshold >=70%).`
    : action === "Watch"
      ? `Choose Watch because the new-product signal does not pass the buying gates: ${credibleAnalogs.length} credible analogs (minimum ${MIN_CREDIBLE_ANALOGS}), ${confidenceScore}% confidence (watch if <55%), or first-12-week forecast ${forecastAvailable ? formatK(selectedTotal) : "N/A"} versus the Watch threshold of ${formatK(watchThreshold)}.`
      : `Choose Reallocate because the XGBoost model predicts ${formatK(selectedTotal)} units for the first 12 weeks, above the Watch threshold of ${formatK(watchThreshold)}, but not strong enough versus the ${draft.category} top-quartile benchmark of ${formatK(benchmark.topQuartile)} for Restock.`;
  const options = data.newProductModel.attributeOptions;
  const forecastPoints = forecastAvailable
    ? [
      { week: 4, value: Math.round(modelPredictions.future4w), total: modelPredictions.future4w, label: "4w" },
      { week: 8, value: Math.round(modelPredictions.future8w), total: modelPredictions.future8w, label: "8w" },
      { week: 12, value: Math.round(modelPredictions.future12w), total: modelPredictions.future12w, label: "12w" },
    ]
    : [];
  const chartLabels = [4, 8, 12];
  const handleImageUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setImageStatus("analyzing");
    setImageMessage("Analyzing image with local LM Studio...");

    try {
      const imageDataUrl = await fileToDataUrl(file);
      setUploadedImage(imageDataUrl);
      const result = await classifyProductImageWithLmStudio({ imageDataUrl, options });
      setDraft({ ...draft, category: result.category, color: result.color, fabric: result.fabric });
      setImageStatus("matched");
      setImageMessage(`Matched ${result.category}, ${result.color}, ${result.fabric}. ${result.reasoning}`);
    } catch (error) {
      setImageStatus("error");
      setImageMessage(error.message || "Could not classify the uploaded image. Please select attributes manually.");
    } finally {
      event.target.value = "";
    }
  };

  return (
    <>
      <header className="topbar">
        <div><span className="caption">New product simulator</span><h1>Attribute-based XGBoost forecast</h1><p>Input category, color, and fabric to estimate first 4w, 8w, and 12w demand.</p></div>
        <div className="summary-strip"><div><span>Model</span><strong>XGBoost</strong></div><div><span>Credible matches</span><strong>{credibleAnalogs.length}</strong></div><div><span>Forecast</span><strong>{forecastAvailable ? "4w / 8w / 12w" : "N/A"}</strong></div></div>
      </header>
      <section className="new-product-grid">
        <div className="controls panel new-product-attributes">
          <div className="panel-heading compact"><div><h2>New product attributes</h2><p>Select planned product attributes. The trained model predicts first 4w, 8w, and 12w demand.</p></div><LineChart size={20} /></div>
          <div className="attribute-grid">
            <AttributeSelect label="Category" value={draft.category} options={options.categories} onChange={(category) => setDraft({ ...draft, category })} />
            <AttributeSelect label="Color" value={draft.color} options={options.colors} onChange={(color) => setDraft({ ...draft, color })} />
            <AttributeSelect label="Fabric" value={draft.fabric} options={options.fabrics} onChange={(fabric) => setDraft({ ...draft, fabric })} />
          </div>
          <div className={`image-match-box ${uploadedImage ? "has-preview" : ""}`}>
            <label className="image-upload-control">
              <span>Upload product image</span>
              <input type="file" accept="image/*" onChange={handleImageUpload} />
            </label>
            {uploadedImage && <img src={uploadedImage} alt="Uploaded new product preview" />}
            <p className={imageStatus}>{imageMessage || "Optional: use local LM Studio to suggest category, color, and fabric from an image."}</p>
          </div>
          <div className="fixed-window new-product-window">
            <div><span>No actual sales yet</span><strong>New product</strong></div>
            <div><span>Model input</span><strong>Category / color / fabric</strong></div>
          </div>
        </div>
        <DemandOutlookPanel
          forecastTotal={selectedTotal}
          categoryMedian={benchmark.median}
          restockEvents={0}
          confidenceScore={confidenceScore}
          intro="XGBoost pre-launch demand signal from category, color, and fabric. No actual sales exist yet for this new product."
          forecastWindow={forecastAvailable ? forecastWindow : "N/A"}
          forecastTooltip={`Projected first-12-week units for a new product. Formula: category, color, and fabric are passed to trained XGBoost regressors that output future4wDemand, future8wDemand, and future12wDemand. The displayed main figure is future12wDemand.`}
          benchmarkTooltip={`Median W0-W11 units sold for ${draft.category} products in the training data. It is on the same first-12-week unit basis as the new-product model output.`}
          restockLabel="Credible analogs"
          restockValue={credibleAnalogs.length}
          restockSubLabel="score >=75"
          restockTooltip="Number of matched analog products with Match score >=75. Formula: category 40 points, color 40 points, fabric 20 points. Fewer than 3 credible analogs caps confidence and should trigger human review."
          forecastDisplay={forecastAvailable ? undefined : "N/A"}
          confidenceTooltip={`New-product confidence uses the held-out XGBoost test accuracy for the 12-week target (${validation?.testAccuracyPct ?? "N/A"}% accuracy, WAPE ${validation?.testWapePct ?? "N/A"}%), then caps the score at Low confidence if fewer than ${MIN_CREDIBLE_ANALOGS} credible analogs exist.`}
          insight={forecastAvailable ? `XGBoost predicts ${formatK(selectedTotal)} first-12-week units vs. ${draft.category} W0-W11 median ${formatK(benchmark.median)} and top quartile ${formatK(benchmark.topQuartile)}.` : `Model forecast is not available for this attribute combination or the model artifact has not loaded.`}
          modelOutputs={forecastAvailable ? [
            { label: "Future 4w", value: formatK(modelPredictions.future4w) },
            { label: "Future 8w", value: formatK(modelPredictions.future8w) },
            { label: "Future 12w", value: formatK(modelPredictions.future12w) },
          ] : null}
        />
        <RecommendationPanel
          action={action}
          confidence={confidence}
          lead={`Prediction uses category, color, and fabric. Matched analogs are shown as evidence and guardrail, not as the forecast engine.`}
          metrics={[
            { label: "Model output", value: forecastAvailable ? "4w / 8w / 12w" : "N/A" },
            { label: "Future 12w units", value: forecastAvailable ? formatK(selectedTotal) : "N/A" },
            { label: "Watch threshold", value: formatK(watchThreshold) },
            { label: "Recommended action", value: action },
            { label: "Key driver", value: credibleAnalogs.length < 3 ? "Low credible analog coverage" : "XGBoost demand + category benchmark" },
          ]}
          detailItems={[
            { label: "Model input", value: `${draft.category} / ${draft.color} / ${draft.fabric}`, note: "The new product has no actual sales, so the model only uses product attributes selected by the planner.", logic: "The trained XGBoost feature set for new products is category, color, fabric, plus smoothed historical group benchmarks learned from the training set. Season is not used." },
            { label: "Model outputs", value: forecastAvailable ? `${formatK(selectedTotal)} future 12w` : "N/A", note: forecastAvailable ? `Future 4w ${formatK(modelPredictions.future4w)}, future 8w ${formatK(modelPredictions.future8w)}, future 12w ${formatK(modelPredictions.future12w)}.` : "Model output is unavailable because the artifact did not contain this attribute key.", logic: "Three XGBoost regressors are trained separately for future4wDemand, future8wDemand, and future12wDemand. Outputs are cumulative demand totals for the first 4, 8, and 12 weeks after launch." },
            { label: "Model accuracy", value: `${validation?.testAccuracyPct ?? "N/A"}%`, note: `Held-out test WAPE is ${validation?.testWapePct ?? "N/A"}%; bias is ${validation?.testBiasPct ?? "N/A"}%.`, logic: "Accuracy is calculated as 1 - WAPE on the 15% product-level test set. The split is product-level 70/15/15, so test products are not seen during training." },
            { label: "Category benchmark", value: `${formatK(benchmark.median)} median`, note: `Compared with ${draft.category} W0-W11 median ${formatK(benchmark.median)} and top quartile ${formatK(benchmark.topQuartile)}.`, logic: "Benchmark uses historical W0-W11 launch demand for the same category, so it is directly comparable with the model's future12wDemand output." },
            { label: "Watch threshold", value: `${formatK(watchThreshold)} first 12w`, note: `Calculated as 75% of the ${draft.category} W0-W11 median (${formatK(benchmark.median)}).`, logic: "Watch threshold = category median x 0.75. It marks the low-demand zone for a new product: if predicted first-12-week demand is below this value, the product is materially weaker than the historical category baseline. The threshold is a review trigger, not an automatic no-buy rule or an order quantity." },
            { label: "Analog guardrail", value: `${credibleAnalogs.length} credible analogs`, note: credibleAnalogs.length < 3 ? "Fewer than 3 credible analogs caps confidence and should be escalated for human review." : "Analog coverage is sufficient as supporting evidence.", logic: `Credible analogs must score at least 75. Match formula for new products: category 40 points, color 40 points, fabric 20 points. The analogs explain similarity; they do not calculate the forecast.` },
            { label: "Decision rule", value: action, note: "The action is based on model demand, category benchmark, confidence, and credible analog coverage.", logic: "Restock: future12wDemand >= category W0-W11 top quartile, confidence >=70%, and at least 3 credible analogs. Watch: fewer than 3 credible analogs, confidence <55%, or future12wDemand below the Watch threshold, where Watch threshold = 75% of category median. Reallocate: middle signal." },
          ]}
          rationaleText={rationaleText}
          discountRisk={false}
        />
      </section>
      <section className="main-grid single-chart">
        {forecastAvailable ? (
          <TrendChart
            title="New product 4w / 8w / 12w forecast"
            subtitle="No actual sales yet. Dots and tiles show cumulative model demand for each horizon."
            labels={chartLabels}
            actualPoints={[]}
            forecastPoints={forecastPoints}
            bandPoints={[]}
            highlightWeeks={chartLabels}
            weeklyTiles={forecastPoints.map((point) => ({ label: `Future ${point.label} total`, week: point.week, value: point.total, kind: "forecast-tile" }))}
            totalLabel="Future 12w total"
            totalValue={selectedTotal}
            forecastTooltip="XGBoost outputs cumulative future4wDemand, future8wDemand, and future12wDemand from category, color, and fabric. Chart dots and tiles show the same cumulative totals."
            analogRangeTooltip="Analog range is not shown on this forecast chart because the new-product forecast comes from XGBoost model outputs, not an analog weekly curve."
          />
        ) : (
          <section className="panel no-forecast-panel">
            <div>
              <AlertTriangle size={20} />
              <h2>Forecast not available</h2>
              <p>The XGBoost prediction artifact is missing this category, color, and fabric key. Confirm the model file has been generated before using this combination.</p>
            </div>
          </section>
        )}
      </section>
      <section className="new-product-lower-grid"><AnalogCards analogs={credibleAnalogs} compactScroll enablePopup /><Benchmark label={`${draft.category} W0-W11 category benchmark across ${benchmark.count} products.`} values={[{ label: forecastAvailable ? "Future 12w model" : "Forecast unavailable", value: forecastAvailable ? selectedTotal : 0, kind: "product" }, { label: "Category top quartile", value: benchmark.topQuartile, kind: "top" }, { label: "Category median", value: benchmark.median, kind: "median" }]} /></section>
    </>
  );
}

function App() {
  const [data, setData] = useState(null);
  const [existingModel, setExistingModel] = useState(null);
  const [newProductModelArtifact, setNewProductModelArtifact] = useState(null);
  const [mode, setMode] = useState("existing");
  const [selectedId, setSelectedId] = useState(null);
  const [draft, setDraft] = useState({ category: "culottes", color: "black", fabric: "bengaline" });

  useEffect(() => {
    Promise.all([
      fetch("/data/demo-data.json").then((res) => res.json()),
      fetch("/data/existing-demand-model.json").then((res) => res.ok ? res.json() : null).catch(() => null),
      fetch("/data/new-product-demand-model.json").then((res) => res.ok ? res.json() : null).catch(() => null),
    ]).then(([payload, model, newModel]) => {
      setData(payload);
      setExistingModel(model);
      setNewProductModelArtifact(newModel);
      setSelectedId(payload.products[0].id);
      const o = payload.newProductModel.attributeOptions;
      setDraft({ category: o.categories[0], color: o.colors[0], fabric: o.fabrics[0] });
    });
  }, []);

  const product = useMemo(() => data?.products.find((item) => item.id === selectedId), [data, selectedId]);
  const dataAsOf = data?.summary?.dataAsOf ? formatDate(data.summary.dataAsOf) : "N/A";

  if (!data || !product) return <div className="loading">Loading Visuelle demand signals...</div>;

  return (
    <div className="app">
      <ProductSelector products={data.products} selectedId={selectedId} setSelectedId={(id) => { setSelectedId(id); setMode("existing"); }} />
      <main className="workspace">
        <div className="app-topline">
          <div className="mode-tabs">
            <button className={mode === "existing" ? "active" : ""} onClick={() => setMode("existing")}>Existing product</button>
            <button className={mode === "new" ? "active" : ""} onClick={() => setMode("new")}>New product</button>
          </div>
          <div className="utility-bar">
            <span>Data as of: {dataAsOf}</span>
            <HelpCircle size={17} />
            <Bell size={17} />
            <strong>LP</strong>
            <span>Leo Planner</span>
          </div>
        </div>
        {mode === "existing" ? (
          <ExistingProductView data={data} product={product} existingModel={existingModel} />
        ) : (
          <NewProductView data={data} draft={draft} setDraft={setDraft} newProductModelArtifact={newProductModelArtifact} />
        )}
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
