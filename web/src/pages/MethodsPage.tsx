import { useQueries } from "@tanstack/react-query";
import { CheckCircle2, CircleAlert } from "lucide-react";

import { EvidenceDisclosure } from "../components/EvidenceDisclosure";
import { JsonTable } from "../components/JsonTable";
import { ErrorBlock, LoadingBlock } from "../components/Status";
import { api } from "../lib/api";
import { formatNumber, sentenceCase } from "../lib/format";
import type { AnyRecord } from "../lib/types";

export function MethodsPage() {
  const [health, evaluation] = useQueries({
    queries: [
      { queryKey: ["health"], queryFn: api.health },
      { queryKey: ["evaluation"], queryFn: api.evaluation }
    ]
  });

  if (health.isLoading || evaluation.isLoading) {
    return <LoadingBlock label="Loading methods and readiness" />;
  }
  if (health.isError) {
    return <ErrorBlock title="Could not load health" detail={health.error.message} />;
  }
  if (evaluation.isError) {
    return <ErrorBlock title="Could not load evaluation" detail={evaluation.error.message} />;
  }

  return (
    <section className="page-grid">
      <div className="page-heading">
        <p className="eyebrow">Methods, evaluation, reproducibility</p>
        <h1>Grounded discovery over table artifacts.</h1>
        <p>
          StatVocab indexes table titles, extracted vocabulary, geography, time, domains, and
          evidence. It returns ranked source tables only, never unsupported numeric answers.
        </p>
      </div>

      <section className="two-column">
        <article className="panel">
          <h2>Backend Health</h2>
          <div className={`health-banner health-${health.data?.status}`}>
            {health.data?.status === "ok" ? (
              <CheckCircle2 aria-hidden="true" size={18} />
            ) : (
              <CircleAlert aria-hidden="true" size={18} />
            )}
            <strong>{health.data?.status}</strong>
            <span>{health.data?.document_count.toLocaleString()} indexed documents</span>
          </div>
          <JsonTable
            record={{
              config_name: health.data?.config_name,
              corpus: health.data?.corpus,
              loaded_search_run_id: health.data?.loaded_search_run_id
            }}
          />
        </article>
        <article className="panel">
          <h2>Artifact Counts</h2>
          <JsonTable record={health.data?.artifact_counts} />
        </article>
      </section>

      <section className="panel">
        <h2>Readiness</h2>
        <div className="readiness-grid">
          {Object.entries(health.data?.readiness || {}).map(([key, ready]) => (
            <span className={`readiness readiness-${ready ? "ready" : "missing"}`} key={key}>
              {ready ? <CheckCircle2 aria-hidden="true" size={16} /> : <CircleAlert aria-hidden="true" size={16} />}
              {sentenceCase(key)}
            </span>
          ))}
        </div>
        {health.data?.warnings.length ? (
          <ul className="plain-list">
            {health.data.warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        ) : null}
      </section>

      <section className="panel">
        <h2>Evaluation Summaries</h2>
        <div className="evaluation-grid">
          {Object.entries(evaluation.data?.summaries || {}).map(([area, summary]) => (
            <article className="evaluation-card" key={area}>
              <div className="result-title-row">
                <h3>{sentenceCase(area)}</h3>
                <span className={`status-pill status-${summary.status}`}>{summary.status}</span>
              </div>
              <MetricSummary record={summary.data} />
              <EvidenceDisclosure
                title="Full evaluation artifact"
                summary={summary.path}
                value={summary.data}
              />
            </article>
          ))}
        </div>
      </section>
    </section>
  );
}

function MetricSummary({ record }: { record?: AnyRecord }) {
  const metrics = collectMetrics(record);
  if (!metrics.length) {
    return <p className="muted">No headline metrics reported.</p>;
  }

  return (
    <dl className="metric-grid">
      {metrics.map(([key, value]) => (
        <div key={key}>
          <dt>{key}</dt>
          <dd>{formatMetric(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function collectMetrics(record?: AnyRecord): Array<[string, number]> {
  const wanted = [
    "HitRate@10",
    "HitRate@5",
    "MRR",
    "macro_f1",
    "accuracy",
    "coverage",
    "relation_count",
    "cluster_count",
    "question_count"
  ];
  const found = new Map<string, number>();

  function visit(value: unknown) {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      return;
    }
    for (const [key, child] of Object.entries(value as AnyRecord)) {
      if (wanted.includes(key) && typeof child === "number" && Number.isFinite(child)) {
        found.set(key, child);
      }
      visit(child);
    }
  }

  visit(record);
  return wanted
    .filter((key) => found.has(key))
    .slice(0, 5)
    .map((key) => [key, found.get(key) as number]);
}

function formatMetric(value: number): string {
  if (value > 0 && value <= 1) {
    return `${Math.round(value * 100)}%`;
  }
  return formatNumber(value);
}
