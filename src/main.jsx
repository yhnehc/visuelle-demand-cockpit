import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AlertTriangle,
  Bell,
  ArrowRightLeft,
  Boxes,
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

const EXISTING_FUTURE_WEEKS = Array.from({ length: 12 }, (_, index) => index + 12);
const NEW_PRODUCT_WEEKS = Array.from({ length: 12 }, (_, index) => index);
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
  const analogTotals = product.analogs?.map((item) => item.total12).filter((value) => Number.isFinite(value)) || [];
  const analogMedian = quantile(analogTotals, 0.5);
  const analogIqr = quantile(analogTotals, 0.75) - quantile(analogTotals, 0.25);
  const consistency = analogMedian ? Math.max(0, 1 - Math.min(1.2, analogIqr / analogMedian) / 1.2) : 0;
  const actualHistoryScore = product.weekSales?.length >= 12 ? 25 : 0;
  const analogCoverageScore = Math.min(20, analogTotals.length * 4);
  const consistencyScore = consistency * 25;
  const discountScore = product.avgDiscount <= 0.1 ? 15 : product.avgDiscount <= 0.2 ? 11 : 7;
  const storeCoverageScore = product.stores >= 80 ? 15 : product.stores >= 50 ? 11 : 7;

  return Math.max(35, Math.min(92, Math.round(
    actualHistoryScore + analogCoverageScore + consistencyScore + discountScore + storeCoverageScore
  )));
}

function calculateNewProductConfidence(analogs) {
  const totals = analogs.map((item) => item.total12).filter((value) => Number.isFinite(value));
  const analogMedian = quantile(totals, 0.5);
  const analogIqr = quantile(totals, 0.75) - quantile(totals, 0.25);
  const consistency = analogMedian ? Math.max(0, 1 - Math.min(1.2, analogIqr / analogMedian) / 1.2) : 0;
  const analogCoverageScore = Math.min(40, totals.length * 8);
  const consistencyScore = consistency * 35;
  const scopePenalty = totals.length < 3 ? -15 : 0;

  return Math.max(25, Math.min(85, Math.round(20 + analogCoverageScore + consistencyScore + scopePenalty)));
}

function classForSignal(signal) {
  if (signal.includes("Restock")) return "positive";
  if (signal.includes("transfer")) return "risk";
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
    .map((item) => item.total12)
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

function pathFor(points, xFor, yFor) {
  return points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${xFor(point.week).toFixed(1)} ${yFor(point.value).toFixed(1)}`)
    .join(" ");
}

function matchAnalogs(catalog, draft) {
  return [...catalog]
    .map((item) => {
      const score =
        (item.category === draft.category ? 5 : 0) +
        (item.color === draft.color ? 2 : 0) +
        (item.fabric === draft.fabric ? 3 : 0);
      return { ...item, matchScore: score };
    })
    .filter((item) => item.matchScore > 0)
    .sort((a, b) => b.matchScore - a.matchScore || b.total12 - a.total12)
    .slice(0, 5);
}

function credibleNewProductAnalogs(analogs) {
  return analogs.filter((item) => (item.matchScore || 0) * 10 >= 75);
}

function newProductCategoryBenchmark(catalog, draft) {
  const categoryTotals = catalog
    .filter((item) => item.category === draft.category)
    .map((item) => item.total12)
    .filter((value) => Number.isFinite(value));

  return {
    median: Math.round(quantile(categoryTotals, 0.5)),
    topQuartile: Math.round(quantile(categoryTotals, 0.75)),
    count: categoryTotals.length,
  };
}

function decideNewProductAction({ forecastTotal, confidenceScore, credibleCount, benchmark }) {
  const weakAnalogSupport = credibleCount < 3;
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
    .sort((a, b) => b.matchScore - a.matchScore || b.total12 - a.total12)
    .slice(0, limit);
}

function forecastFromAnalogs(analogs) {
  if (!analogs.length) return Array(12).fill(0);
  return Array.from({ length: 12 }, (_, week) =>
    Math.round(analogs.reduce((sum, item) => sum + item.weekSales[week], 0) / analogs.length)
  );
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
  forecastTooltip = "Projected future unit sales after the actual sales window. For existing products, W12-W23 is estimated from the observed W0-W11 sales curve and matched historical analog evidence.",
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
      <div className={`weekly-actuals ${tileMode === "twelve" ? "twelve-tiles" : ""}`}>
        {weeklyTiles.map((tile) => (
          <div key={tile.label} className={tile.kind}>
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
  const opportunity = action === "Restock" ? "High demand opportunity" : action === "Watch" ? "Review before buying" : "Allocation opportunity";
  const fallbackDecisionCopy = {
    Restock: "Choose Restock when evidence confidence is at least 70%, forecasted demand is above the category baseline, and discount pressure is not the main reason for sales.",
    Watch: "Choose Watch when evidence confidence is below 70%, forecasted demand is small, analog support is weak, or discount pressure may be inflating sales.",
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
  categoryMedian,
  restockEvents,
  confidenceScore,
  delta = 0,
  intro = "Next 12-week demand signal based on actual sales and matched analogs.",
  forecastWindow = "W12-W23",
  forecastTooltip = "Projected W12-W23 units. For existing products, this begins after the real W0-W11 sales window and is derived from the product curve plus matched analog evidence.",
  benchmarkTooltip = "Median W0-W11 units sold for products in the same category. It is used as a baseline to judge whether this item is above or below typical category demand.",
  restockLabel = "Restock signal",
  restockValue,
  restockSubLabel,
  restockTooltip = "Count of historical restock events available in the Visuelle data for this product. It is treated as a directional replenishment signal, not current stock on hand.",
  insight,
}) {
  const confidence = confidenceLabel(confidenceScore);
  const tooltips = {
    forecast: forecastTooltip,
    benchmark: benchmarkTooltip,
    restock: restockTooltip,
    confidence: "Evidence confidence score, not forecast accuracy. Formula: actual-history coverage + analog coverage + analog consistency + discount cleanliness + store coverage, capped between 35% and 92%.",
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
          <span><TooltipLabel label="Forecasted units" body={tooltips.forecast} /></span>
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
        <Boxes size={22} />
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
  const canOpenAnalog = enablePopup || !!product?.weekSales;

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
                <div className="analog-metrics"><span>Actual W0-W11</span><strong>{formatK(item.total12)}</strong></div>
                <div className="match-score">
                  <span><TooltipLabel label="Match score" body="Similarity to the selected product. Formula: same category = 40 points, same color = 20, same fabric = 25, same season = 15. Maximum score is 100%." /></span>
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
  const labels = Array.from({ length: 12 }, (_, week) => week);
  const actualPoints = analog.weekSales.map((value, week) => ({ week, value }));
  const comparisonPoints = product?.weekSales ? product.weekSales.map((value, week) => ({ week, value })) : [];
  const analogTop = Array.from({ length: 12 }, (_, week) => ({
    week,
    value: Math.round(quantile(analogs.map((item) => item.weekSales?.[week] ?? 0), 0.75)),
  }));
  const analogBottom = Array.from({ length: 12 }, (_, week) => ({
    week,
    value: Math.round(quantile(analogs.map((item) => item.weekSales?.[week] ?? 0), 0.25)),
  }));
  const total = analog.weekSales.reduce((sum, value) => sum + value, 0);
  const peakValue = Math.max(...analog.weekSales);
  const peakWeek = analog.weekSales.indexOf(peakValue);
  const matchScore = product ? analogMatchScore(product, analog) : Math.min(100, Math.round((analog.matchScore || 0) * 10));

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
              <div><dt>Total W0-W11</dt><dd>{formatK(total)}</dd></div>
              <div><dt>Peak week</dt><dd>W{peakWeek}</dd></div>
              <div><dt>Peak units</dt><dd>{formatK(peakValue)}</dd></div>
              <div><dt>Match score</dt><dd>{matchScore}%</dd></div>
            </dl>
          </div>
        </div>
        <TrendChart
          title="Actual historical sales"
          subtitle={product ? `Observed W0-W11 unit sales for analog product ${analog.id}, compared with current product ${product.id} and the matched analog range.` : `Observed W0-W11 unit sales for analog product ${analog.id}, compared with the matched analog range.`}
          labels={labels}
          actualPoints={actualPoints}
          forecastPoints={[]}
          bandPoints={[analogTop, analogBottom]}
          highlightWeeks={[]}
          comparisonPoints={comparisonPoints}
          comparisonLabel={product ? `Current product ${product.id}` : "Current product"}
          analogRangeTooltip="Analog range for W0-W11. For each week, the shaded band is the 25th to 75th percentile of the matched analog products shown in the Historical analogs row."
          weeklyTiles={analog.weekSales.map((value, week) => ({ label: `W${week}`, value, kind: "observed" }))}
          tileMode="twelve"
          totalLabel="Total units"
          totalValue={total}
        />
      </section>
    </div>
  );
}

function StoreSignals({ stores, forecastTotal }) {
  const [isOpen, setIsOpen] = useState(false);
  const displayedStoreTotal = stores.reduce((sum, store) => sum + store.total12, 0);
  const forecastTooltip = `Directional allocation forecast for displayed sample stores only. Formula: product W12-W23 forecast (${formatK(forecastTotal)} units) x store W0-W11 units / displayed-store W0-W11 total (${formatK(displayedStoreTotal)} units). Inventory depth and store profiles are not available, so this is not an exact store order quantity.`;
  const rows = stores.map((store) => ({
    ...store,
    storeForecast: displayedStoreTotal ? Math.round((forecastTotal * store.total12) / displayedStoreTotal) : 0,
  }));

  return (
    <>
      <section className="panel store-panel">
        <div className="panel-heading"><div><h2>Store-level allocation signals</h2><p>Directional signals because inventory depth and store profiles are missing.</p></div><Store size={20} /></div>
        <div className="store-table">
          <button className="view-stores-button" type="button" onClick={() => setIsOpen(true)}>View all stores</button>
          <div className="store-head">
            <span>Store</span>
            <span>W0-W11</span>
            <span><TooltipLabel label="Forecast W12-W23" body={forecastTooltip} /></span>
            <span>Signal</span>
          </div>
          {rows.slice(0, 5).map((store) => (
            <div className="store-row" key={store.store}>
              <strong>{store.store}</strong><span>{store.total12}</span><span>{store.storeForecast}</span><em className={classForSignal(store.signal)}>{store.signal}</em>
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
                <h2 id="store-modal-title">All displayed stores</h2>
              </div>
              <button type="button" aria-label="Close store details" onClick={() => setIsOpen(false)}><X size={18} /></button>
            </div>
            <p className="store-modal-note">{forecastTooltip}</p>
            <div className="store-table modal-store-table">
              <div className="store-head"><span>Store</span><span>W0-W11</span><span>Forecast W12-W23</span><span>Signal</span></div>
              {rows.map((store) => (
                <div className="store-row" key={store.store}>
                  <strong>{store.store}</strong><span>{store.total12}</span><span>{store.storeForecast}</span><em className={classForSignal(store.signal)}>{store.signal}</em>
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

function ExistingProductView({ data, product }) {
  const labels = Array.from({ length: 24 }, (_, i) => i);
  const actualPoints = product.weekSales.map((value, week) => ({ week, value }));
  const forecastPoints = product.futureForecast.map((value, index) => ({ week: index + 12, value }));
  const futureTotal = product.futureForecast.reduce((sum, value) => sum + value, 0);
  const forecastDelta = Math.round(((futureTotal - product.total12) / Math.max(product.total12, 1)) * 100);
  const displayAnalogs = matchHistoricalAnalogs(data.newProductModel.catalog, product, 8);
  const evidenceProduct = { ...product, analogs: displayAnalogs };
  const analogTop = Array.from({ length: 12 }, (_, week) => ({
    week,
    value: Math.round(quantile(displayAnalogs.map((item) => item.weekSales?.[week] ?? 0), 0.75)),
  }));
  const analogBottom = Array.from({ length: 12 }, (_, week) => ({
    week,
    value: Math.round(quantile(displayAnalogs.map((item) => item.weekSales?.[week] ?? 0), 0.25)),
  }));
  const forecastTop = forecastPoints.map((p) => ({ ...p, value: Math.round(p.value * 1.22) }));
  const forecastBottom = forecastPoints.map((p) => ({ ...p, value: Math.round(p.value * 0.78) }));
  const bandTop = [...analogTop, ...forecastTop];
  const bandBottom = [...analogBottom, ...forecastBottom];
  const confidenceScore = calculateEvidenceConfidence(evidenceProduct);
  const categoryBenchmark = categoryBenchmarks(data, product);
  const rationaleText = product.action === "Restock"
    ? `Choose Restock because evidence confidence is ${confidenceScore}% (threshold >=70%), the W12-W23 forecast is ${formatK(futureTotal)} units versus a category median of ${formatK(categoryBenchmark.median)}, restock history shows ${product.restockEvents || 0} events, and average discount is ${Math.round(product.avgDiscount * 100)}% (below the 20% discount-risk threshold).`
    : product.action === "Watch"
      ? `Choose Watch because evidence confidence is ${confidenceScore}% (watch threshold <70%) or the next 12-week forecast is not strong enough versus the category baseline. Average discount is ${Math.round(product.avgDiscount * 100)}%, so the planner should monitor before adding inventory.`
      : `Choose ${product.action} because evidence confidence is ${confidenceScore}%, W12-W23 forecast is ${formatK(futureTotal)} units, and the signal is directional but should be reviewed against store allocation and margin constraints.`;

  return (
    <>
      <header className="topbar">
        <div />
        <div className="summary-strip"><div><span>Dataset</span><strong>{data.summary.productCount.toLocaleString()} products</strong></div><div><span>Stores</span><strong>{data.summary.storeCount}</strong></div><div><span>Actuals</span><strong>W0-W11</strong></div></div>
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
              <div><dt>Week 0-11</dt><dd>{formatK(product.total12)}</dd><small>units</small></div>
              <div><dt>Stores</dt><dd>{product.stores}</dd><small>selling</small></div>
              <div><dt>Avg. discount</dt><dd>{Math.round(product.avgDiscount * 100)}%</dd><small>observed</small></div>
            </dl>
          </div>
        </div>
        <DemandOutlookPanel
          forecastTotal={futureTotal}
          categoryMedian={categoryBenchmark.median}
          restockEvents={product.restockEvents}
          confidenceScore={confidenceScore}
          delta={forecastDelta}
        />
        <RecommendationPanel
          action={product.action}
          confidence={product.confidence}
          lead={`Product ${product.id} already has complete W0-W11 actual sales. Forecast begins at W12.`}
          metrics={[
            { label: "Recommended action", value: product.action },
            { label: "Selected horizon", value: "Next 12 weeks" },
            { label: "Forecasted units", value: `${formatK(futureTotal)} (W12-W23)` },
            { label: "Key driver", value: product.restockEvents ? "Restock + analog match" : "Sales trend + analog match" },
          ]}
          detailItems={[
            { label: "Actual launch sales", value: `${formatK(product.total12)} units`, note: "The product has complete W0-W11 actual sales, so the forecast starts from W12 rather than inventing earlier demand.", logic: "Used to separate real observed demand from future forecast. Strong W0-W11 actuals increase confidence because the product has a complete launch curve." },
            { label: "Future demand", value: `${formatK(futureTotal)} units`, note: "The next 12-week forecast is estimated from the product curve and matched analog evidence.", logic: "Used to decide whether the next selling window is large enough to justify action. This is projected demand, not an exact order quantity." },
            { label: "Historical analogs", value: `${displayAnalogs.length} products with match score >=75`, note: "Analog products share category, color, fabric, or season attributes and are used as directional evidence.", logic: "Only products with Match score of at least 75 are used as matched analog evidence. More credible analogs make the pattern easier to trust; if analogs are too few, the recommendation should be treated as lower confidence." },
            { label: "Discount check", value: `${Math.round(product.avgDiscount * 100)}% avg. discount`, note: product.avgDiscount >= 0.2 ? "Discounting is material, so observed sales should not be treated as pure organic demand." : "Discount pressure is limited, so the observed sales signal is cleaner.", logic: "Discounts can inflate unit sales. Lower average discount means demand is more likely to reflect product appeal rather than markdown pressure." },
            { label: "Store signal", value: product.restockEvents ? `${product.restockEvents} restock events` : "No restock history", note: "Store output is an allocation signal only because inventory depth and full store profiles are missing.", logic: "Restock events indicate replenishment activity in the data. It supports the direction of the recommendation but does not prove current stock level." },
          ]}
          rationaleText={rationaleText}
          discountRisk={product.avgDiscount >= 0.2}
        />
      </section>
      <section className="main-grid single-chart">
        <TrendChart
          title="Actual sales and future forecast"
          subtitle="W0-W11 are real historical sales against the analog range; W12 onward is projected demand."
          labels={labels}
          actualPoints={actualPoints}
          forecastPoints={forecastPoints}
          bandPoints={[bandTop, bandBottom]}
          highlightWeeks={EXISTING_FUTURE_WEEKS}
          weeklyTiles={[...product.weekSales.map((value, week) => ({ label: `W${week}`, value, kind: "observed" })), ...forecastPoints.map((point) => ({ label: `W${point.week}`, value: point.value, kind: "forecast-tile" }))]}
          forecastTooltip="Forecast line for W12-W23. It starts after the real W0-W11 sales window and is projected from the current product's observed sales curve plus matched historical analog evidence. It is demand guidance, not an automatic order quantity."
          analogRangeTooltip="Analog range for W0-W23. For W0-W11, the band is the 25th to 75th percentile of matched analog products with Match score >=75. For W12-W23, the band is a directional uncertainty range around the forecast, using -22% to +22%."
        />
      </section>
      <section className="lower-grid"><StoreSignals stores={product.storeSignals} forecastTotal={futureTotal} /><Benchmark label={`${product.category} actual W0-W11 performance across ${categoryBenchmark.count} category products.`} avgDiscount={product.avgDiscount} values={[{ label: "This product", value: product.total12, kind: "product" }, { label: "Category top quartile", value: categoryBenchmark.topQuartile, kind: "top" }, { label: "Category median", value: categoryBenchmark.median, kind: "median" }]} /><AnalogCards analogs={displayAnalogs} product={product} /></section>
    </>
  );
}

function NewProductView({ data, draft, setDraft }) {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [imageStatus, setImageStatus] = useState("idle");
  const [imageMessage, setImageMessage] = useState("");
  const analogs = useMemo(() => matchAnalogs(data.newProductModel.catalog, draft), [data, draft]);
  const credibleAnalogs = useMemo(() => credibleNewProductAnalogs(analogs), [analogs]);
  const forecastAvailable = credibleAnalogs.length >= 3;
  const forecastAnalogs = forecastAvailable ? credibleAnalogs : [];
  const forecast = useMemo(() => forecastFromAnalogs(forecastAnalogs), [forecastAnalogs]);
  const forecastPoints = forecast.map((value, week) => ({ week, value }));
  const selectedTotal = sumWeeks(forecast, NEW_PRODUCT_WEEKS);
  const totals = forecastAnalogs.map((item) => item.total12);
  const benchmark = newProductCategoryBenchmark(data.newProductModel.catalog, draft);
  const confidenceScore = calculateNewProductConfidence(credibleAnalogs);
  const confidence = confidenceLabel(confidenceScore);
  const action = decideNewProductAction({
    forecastTotal: selectedTotal,
    confidenceScore,
    credibleCount: credibleAnalogs.length,
    benchmark,
  });
  const rationaleText = action === "Restock"
    ? `Choose Restock because predicted W0-W11 launch demand is ${formatK(selectedTotal)} units, above the ${draft.category} top-quartile benchmark of ${formatK(benchmark.topQuartile)}, with ${credibleAnalogs.length} credible analogs (Match score >=75) and ${confidenceScore}% evidence confidence (threshold >=70%).`
    : action === "Watch"
      ? `Choose Watch because the signal is not strong enough for a buying action: ${credibleAnalogs.length} credible analogs (minimum 3), ${confidenceScore}% confidence (watch if <55%), or forecast ${forecastAvailable ? formatK(selectedTotal) : "N/A"} versus the low-demand threshold of ${formatK(Math.round(benchmark.median * 0.75))}.`
      : `Choose Reallocate because predicted W0-W11 launch demand is ${formatK(selectedTotal)} units, with ${credibleAnalogs.length} credible analogs and ${confidenceScore}% confidence, but it is not above the ${draft.category} top-quartile benchmark of ${formatK(benchmark.topQuartile)} with high confidence.`;
  const options = data.newProductModel.attributeOptions;
  const bandTop = forecastPoints.map((p) => ({ ...p, value: Math.round(p.value * 1.25) }));
  const bandBottom = forecastPoints.map((p) => ({ ...p, value: Math.round(p.value * 0.72) }));
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
        <div><span className="caption">New product simulator</span><h1>Attribute-based forecast</h1><p>Match a proposed product to historical analogs and predict W0-W11 launch demand.</p></div>
        <div className="summary-strip"><div><span>Analog pool</span><strong>{data.newProductModel.catalog.length} products</strong></div><div><span>Credible matches</span><strong>{credibleAnalogs.length}</strong></div><div><span>Forecast</span><strong>W0-W11</strong></div></div>
      </header>
      <section className="new-product-grid">
        <div className="controls panel new-product-attributes">
          <div className="panel-heading compact"><div><h2>New product attributes</h2><p>Select planned product attributes. The model finds similar historical products.</p></div><LineChart size={20} /></div>
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
            <div><span>Match basis</span><strong>Category / color / fabric</strong></div>
          </div>
        </div>
        <DemandOutlookPanel
          forecastTotal={selectedTotal}
          categoryMedian={benchmark.median}
          restockEvents={0}
          confidenceScore={confidenceScore}
          intro="12-week demand signal based on credible matched analogs. No actual sales exist yet for this new product."
          forecastWindow="W0-W11"
          forecastTooltip="Projected W0-W11 launch units for a new product. Because no actual sales exist yet, this forecast is the average weekly curve from credible matched analog products with Match score >=75."
          benchmarkTooltip={`Median W0-W11 units sold for ${draft.category} products in the Visuelle catalog. It is used as the category baseline for new product demand.`}
          restockLabel="Credible analogs"
          restockValue={credibleAnalogs.length}
          restockSubLabel="score >=75"
          restockTooltip="Number of matched analog products with Match score >=75. Restock requires at least 3 credible analogs; fewer than 3 should be treated as low-confidence and escalated for human review."
          forecastDisplay={forecastAvailable ? undefined : "N/A"}
          insight={forecastAvailable ? `Forecast is ${formatK(selectedTotal)} vs. ${draft.category} median ${formatK(benchmark.median)} and top quartile ${formatK(benchmark.topQuartile)}.` : `Insufficient credible analogs: ${credibleAnalogs.length} matched products with Match score >=75. Forecast is not shown; escalate for human review.`}
        />
        <RecommendationPanel
          action={action}
          confidence={confidence}
          lead={`Forecast is based on ${credibleAnalogs.length} credible historical analogs with Match score >=75 for ${draft.category}, ${draft.color}, and ${draft.fabric}.`}
          metrics={[
            { label: "Forecast window", value: "W0-W11" },
            { label: "Forecasted units", value: forecastAvailable ? formatK(selectedTotal) : "N/A" },
            { label: "Recommended action", value: action },
            { label: "Key driver", value: credibleAnalogs.length < 3 ? "Low credible analog coverage" : `${draft.category} benchmark + credible analogs` },
          ]}
          detailItems={[
            { label: "Forecast basis", value: `${credibleAnalogs.length} credible analogs`, note: "The new product has no actual sales, so launch demand is inferred from products with Match score >=75.", logic: "Credible analogs must score at least 75 based on category, color, and fabric similarity. If fewer than 3 credible analogs exist, the recommendation is Watch / human review." },
            { label: "Forecasted launch demand", value: forecastAvailable ? `${formatK(selectedTotal)} units` : "N/A", note: forecastAvailable ? `Compared with ${draft.category} median ${formatK(benchmark.median)} and top quartile ${formatK(benchmark.topQuartile)}.` : "Forecast is suppressed because fewer than 3 credible analogs are available.", logic: "Forecast size is evaluated against the selected category's historical W0-W11 distribution, not a fixed 800-unit threshold. If fewer than 3 credible analogs exist, no forecast is shown." },
            { label: "Analog quality", value: confidence, note: credibleAnalogs.length < 3 ? "Fewer than 3 credible analogs means this should be escalated for human review." : "Analog coverage is sufficient for a directional planning signal.", logic: "Confidence is based on credible analog coverage and consistency. Low confidence blocks Restock even when forecasted units look high." },
            { label: "Decision rule", value: action, note: "The action is based on forecast strength, category benchmark, confidence, and credible analog coverage.", logic: "Restock: forecast >= category top quartile, confidence >=70%, and at least 3 credible analogs. Watch: fewer than 3 credible analogs, confidence <55%, or forecast below 75% of category median. Reallocate: middle signal." },
          ]}
          rationaleText={rationaleText}
          discountRisk={false}
        />
      </section>
      <section className="main-grid single-chart">
        {forecastAvailable ? (
          <TrendChart
            title="New product W0-W11 forecast"
            subtitle="No actual sales yet. Forecast is the average weekly curve from credible matched analog products."
            labels={Array.from({ length: 12 }, (_, i) => i)}
            actualPoints={[]}
            forecastPoints={forecastPoints}
            bandPoints={[bandTop, bandBottom]}
            highlightWeeks={NEW_PRODUCT_WEEKS}
            weeklyTiles={forecast.map((value, week) => ({ label: `W${week}`, value, kind: "forecast-tile" }))}
            tileMode="twelve"
          />
        ) : (
          <section className="panel no-forecast-panel">
            <div>
              <AlertTriangle size={20} />
              <h2>Forecast not available</h2>
              <p>Only {credibleAnalogs.length} credible analogs were found with Match score of at least 75. At least 3 credible analogs are required before showing a W0-W11 forecast.</p>
            </div>
          </section>
        )}
      </section>
      <section className="new-product-lower-grid"><AnalogCards analogs={credibleAnalogs} compactScroll enablePopup /><Benchmark label={`${draft.category} W0-W11 category benchmark across ${benchmark.count} products.`} values={[{ label: forecastAvailable ? "Forecast total" : "Forecast unavailable", value: forecastAvailable ? sumWeeks(forecast, Array.from({ length: 12 }, (_, i) => i)) : 0, kind: "product" }, { label: "Category top quartile", value: benchmark.topQuartile, kind: "top" }, { label: "Category median", value: benchmark.median, kind: "median" }]} /></section>
    </>
  );
}

function App() {
  const [data, setData] = useState(null);
  const [mode, setMode] = useState("existing");
  const [selectedId, setSelectedId] = useState(null);
  const [draft, setDraft] = useState({ category: "culottes", color: "black", fabric: "bengaline" });

  useEffect(() => {
    fetch("/data/demo-data.json").then((res) => res.json()).then((payload) => {
      setData(payload);
      setSelectedId(payload.products[0].id);
      const o = payload.newProductModel.attributeOptions;
      setDraft({ category: o.categories[0], color: o.colors[0], fabric: o.fabrics[0] });
    });
  }, []);

  const product = useMemo(() => data?.products.find((item) => item.id === selectedId), [data, selectedId]);

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
            <span>Data as of: May 10, 2025</span>
            <HelpCircle size={17} />
            <Bell size={17} />
            <strong>LP</strong>
            <span>Leo Planner</span>
          </div>
        </div>
        {mode === "existing" ? (
          <ExistingProductView data={data} product={product} />
        ) : (
          <NewProductView data={data} draft={draft} setDraft={setDraft} />
        )}
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
