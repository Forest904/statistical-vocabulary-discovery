import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";

import { CategoryBadge, ConfidenceBadge } from "../components/Badges";
import { EvidenceDisclosure } from "../components/EvidenceDisclosure";
import { JsonTable } from "../components/JsonTable";
import { ErrorBlock, LoadingBlock } from "../components/Status";
import { api } from "../lib/api";
import { compactJson, rawText, relationTypeLabels, sentenceCase } from "../lib/format";

export function TermDetailPage() {
  const { termId = "" } = useParams();
  const term = useQuery({
    queryKey: ["term", termId],
    queryFn: () => api.term(termId),
    enabled: Boolean(termId)
  });

  if (term.isLoading) {
    return <LoadingBlock label="Loading term" />;
  }
  if (term.isError) {
    return <ErrorBlock title="Could not load term" detail={term.error.message} />;
  }
  if (!term.data) {
    return null;
  }

  const data = term.data;
  return (
    <section className="page-grid">
      <div className="page-heading">
        <p className="eyebrow">Vocabulary term</p>
        <h1>{data.canonical_term}</h1>
        <div className="meta-strip">
          <span>{data.term_id}</span>
          <span>{data.table_appearances.length.toLocaleString()} table appearances</span>
          <span>{data.occurrence_ids.length.toLocaleString()} occurrences</span>
        </div>
        <div className="chip-list">
          <CategoryBadge category={data.category} />
          <ConfidenceBadge value={data.confidence} />
        </div>
      </div>

      <section className="two-column">
        <article className="panel">
          <h2>Provenance</h2>
          <JsonTable record={data.provenance} />
        </article>
        <article className="panel">
          <h2>Cluster</h2>
          {data.cluster ? <JsonTable record={data.cluster} /> : <p className="muted">No cluster membership.</p>}
        </article>
      </section>

      <section className="panel">
        <div className="section-title">
          <h2>Table appearances</h2>
          <span>{data.table_appearances.length.toLocaleString()} records</span>
        </div>
        <div className="chip-list">
          {data.table_appearances.slice(0, 80).map((appearance, index) => {
            const tableId = rawText(appearance, ["table_id"]);
            return tableId ? (
              <Link className="simple-chip" to={`/tables/${encodeURIComponent(tableId)}`} key={tableId}>
                {tableId}
              </Link>
            ) : (
              <span className="simple-chip" key={index}>
                {compactJson(appearance)}
              </span>
            );
          })}
        </div>
      </section>

      <section className="panel">
        <h2>Relations</h2>
        {Object.entries(data.relations).map(([direction, relations]) => (
          <div className="relation-section" key={direction}>
            <h3>{sentenceCase(direction)}</h3>
            {relations.length ? (
              relations.map((relation) => (
                <article className="relation-card" key={rawText(relation, ["relation_id"])}>
                  <strong>{relationTypeLabels[rawText(relation, ["relation_type"])] || rawText(relation, ["relation_type"])}</strong>
                  <span>
                    {rawText(relation, ["source_term"])} to {rawText(relation, ["target_term"])}
                  </span>
                  <EvidenceDisclosure title="Relation evidence" value={relation.evidence} />
                </article>
              ))
            ) : (
              <p className="muted">No {direction} relations.</p>
            )}
          </div>
        ))}
      </section>
    </section>
  );
}
