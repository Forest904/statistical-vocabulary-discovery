import { useQuery } from "@tanstack/react-query";
import { ExternalLink, Share2 } from "lucide-react";
import { Link, useParams } from "react-router-dom";

import { CategoryBadge } from "../components/Badges";
import { EvidenceDisclosure } from "../components/EvidenceDisclosure";
import { JsonTable } from "../components/JsonTable";
import { ErrorBlock, LoadingBlock } from "../components/Status";
import { api } from "../lib/api";
import { compactJson, formatNumber, rawText, recordCountBy, sentenceCase, uniqueTexts } from "../lib/format";

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
  const geographies = uniqueTexts(table.geographies, ["name", "code", "raw_value"], 28);
  const times = uniqueTexts(table.times, ["normalized_value", "raw_value"], 28);
  const evidenceCounts = recordCountBy(table.evidence, "kind");
  return (
    <section className="page-grid">
      <div className="page-heading">
        <p className="eyebrow">Table evidence</p>
        <h1>{table.title}</h1>
        <div className="meta-strip">
          <span>{table.table_id}</span>
          <span>Parsed: {table.parse_status}</span>
          <span>{formatNumber(table.metadata?.observation_count)} observations</span>
        </div>
        {table.source_url ? (
          <a className="inline-action" href={table.source_url} target="_blank" rel="noreferrer">
            <ExternalLink aria-hidden="true" size={16} />
            Open source table
          </a>
        ) : null}
        <Link
          className="inline-action"
          to={`/graph?focus_type=table&focus_id=${encodeURIComponent(table.table_id)}`}
        >
          <Share2 aria-hidden="true" size={16} />
          Open in graph
        </Link>
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
              <div className="subsection-title">
                <h3>{sentenceCase(category)}</h3>
                <CategoryBadge category={categoryForGroup(category)} />
              </div>
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
          <div className="section-title">
            <h2>Geographies</h2>
            <span>{table.geographies.length.toLocaleString()} records</span>
          </div>
          <div className="chip-list">
            {geographies.map((item) => (
              <span className="simple-chip" key={item}>
                {item}
              </span>
            ))}
            {table.geographies.length > geographies.length ? (
              <span className="simple-chip">+{table.geographies.length - geographies.length} more records</span>
            ) : null}
          </div>
        </article>
        <article className="panel">
          <div className="section-title">
            <h2>Times</h2>
            <span>{table.times.length.toLocaleString()} records</span>
          </div>
          <div className="chip-list">
            {times.map((item) => (
              <span className="simple-chip" key={item}>
                {item}
              </span>
            ))}
          </div>
        </article>
      </section>

      <section className="panel">
        <div className="section-title">
          <h2>Evidence</h2>
          <span>{table.evidence.length.toLocaleString()} records</span>
        </div>
        <div className="chip-list evidence-counts">
          {Object.entries(evidenceCounts).map(([kind, count]) => (
            <span className="simple-chip" key={kind}>
              {sentenceCase(kind)}: {count.toLocaleString()}
            </span>
          ))}
        </div>
        <EvidenceDisclosure
          title="Raw evidence records"
          summary={`${table.evidence.length.toLocaleString()} records, preserved for audit`}
        >
          <div className="evidence-list">
            {table.evidence.slice(0, 80).map((item, index) => (
              <pre key={`${rawText(item, ["evidence_id", "raw_value"])}-${index}`}>
                {compactJson(item)}
              </pre>
            ))}
          </div>
        </EvidenceDisclosure>
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

function categoryForGroup(category: string): string {
  if (category === "other_ambiguous") {
    return category;
  }
  if (category === "domains") {
    return "domain";
  }
  return category.endsWith("s") ? category.slice(0, -1) : category;
}
