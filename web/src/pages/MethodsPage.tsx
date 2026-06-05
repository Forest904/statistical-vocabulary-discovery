import { useQueries } from "@tanstack/react-query";
import { CheckCircle2, CircleAlert } from "lucide-react";

import { JsonTable } from "../components/JsonTable";
import { ErrorBlock, LoadingBlock } from "../components/Status";
import { api } from "../lib/api";
import { compactJson, sentenceCase } from "../lib/format";

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
              <p className="muted">{summary.path}</p>
              <pre>{compactJson(summary.data)}</pre>
            </article>
          ))}
        </div>
      </section>
    </section>
  );
}
