import { useQuery } from "@tanstack/react-query";
import { ExternalLink } from "lucide-react";
import { Link, useParams } from "react-router-dom";

import { CategoryBadge } from "../components/Badges";
import { JsonTable } from "../components/JsonTable";
import { ErrorBlock, LoadingBlock } from "../components/Status";
import { api } from "../lib/api";
import { compactJson, rawText, sentenceCase } from "../lib/format";

export function TableDetailPage() {
  const { tableId = "" } = useParams();
  const query = useQuery({
    queryKey: ["table", tableId],
    queryFn: () => api.table(tableId),
    enabled: Boolean(tableId)
  });

  if (query.isLoading) {
    return <LoadingBlock label="Loading table evidence" />;
  }
  if (query.isError) {
    return <ErrorBlock title="Could not load table" detail={query.error.message} />;
  }
  if (!query.data) {
    return null;
  }

  const table = query.data;
  return (
    <section className="page-grid">
      <div className="page-heading">
        <p className="eyebrow">Table evidence</p>
        <h1>{table.title}</h1>
        <p>
          {table.table_id} · Parse status: <strong>{table.parse_status}</strong>
        </p>
        {table.source_url ? (
          <a className="inline-action" href={table.source_url} target="_blank" rel="noreferrer">
            <ExternalLink aria-hidden="true" size={16} />
            Open source table
          </a>
        ) : null}
      </div>

      <section className="two-column">
        <article className="panel">
          <h2>Metadata</h2>
          <JsonTable record={table.metadata} />
        </article>
        <article className="panel">
          <h2>Vocabulary</h2>
          {Object.entries(table.vocabulary).map(([category, values]) => (
            <div className="vocab-group" key={category}>
              <h3>
                <CategoryBadge category={category.replace(/s$/, "")} />
                {sentenceCase(category)}
              </h3>
              <div className="chip-list">
                {(values || []).slice(0, 40).map((value, index) => (
                  <span className="simple-chip" key={`${String(value)}-${index}`}>
                    {String(value)}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </article>
      </section>

      <section className="two-column">
        <article className="panel">
          <h2>Geographies</h2>
          <ul className="plain-list">
            {table.geographies.slice(0, 40).map((item, index) => (
              <li key={`${rawText(item, ["geography_id", "code", "name"])}-${index}`}>
                {rawText(item, ["name", "code", "raw_value"]) || compactJson(item)}
              </li>
            ))}
          </ul>
        </article>
        <article className="panel">
          <h2>Times</h2>
          <ul className="plain-list">
            {table.times.slice(0, 40).map((item, index) => (
              <li key={`${rawText(item, ["time_id", "normalized_value"])}-${index}`}>
                {rawText(item, ["normalized_value", "raw_value"]) || compactJson(item)}
              </li>
            ))}
          </ul>
        </article>
      </section>

      <section className="panel">
        <h2>Evidence</h2>
        <div className="evidence-list">
          {table.evidence.slice(0, 80).map((item, index) => (
            <pre key={`${rawText(item, ["evidence_id", "raw_value"])}-${index}`}>
              {compactJson(item)}
            </pre>
          ))}
        </div>
      </section>

      <section className="panel">
        <h2>Related Terms</h2>
        <div className="chip-list">
          {table.related_terms.slice(0, 80).map((item, index) => {
            const termId = rawText(item, ["term_id"]);
            return termId ? (
              <Link className="simple-chip" to={`/terms/${encodeURIComponent(termId)}`} key={termId}>
                {rawText(item, ["canonical_term"]) || termId}
              </Link>
            ) : (
              <span className="simple-chip" key={index}>
                {compactJson(item)}
              </span>
            );
          })}
        </div>
      </section>
    </section>
  );
}
