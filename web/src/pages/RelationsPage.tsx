import { useQuery } from "@tanstack/react-query";
import { Filter } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";

import { ConfidenceBadge } from "../components/Badges";
import { PaginationControls } from "../components/Pagination";
import { EmptyBlock, ErrorBlock, LoadingBlock } from "../components/Status";
import { api } from "../lib/api";
import { compactJson, rawText, relationTypeLabels } from "../lib/format";
import type { RelationType } from "../lib/types";

const relationOptions: Array<RelationType | "all"> = [
  "all",
  "broader_than",
  "narrower_than",
  "variant_of",
  "related_to"
];

export function RelationsPage() {
  const [relationType, setRelationType] = useState<RelationType | "all">("all");
  const [termId, setTermId] = useState("");
  const [submittedTermId, setSubmittedTermId] = useState("");
  const [page, setPage] = useState(1);
  const relations = useQuery({
    queryKey: ["relations", relationType, submittedTermId, page],
    queryFn: () =>
      api.relations({
        relation_type: relationType,
        term_id: submittedTermId,
        page,
        page_size: 20
      })
  });

  return (
    <section className="page-grid">
      <div className="page-heading">
        <p className="eyebrow">Bounded relationship explorer</p>
        <h1>Review cautious measure-to-measure candidates.</h1>
      </div>

      <div className="toolbar">
        <div className="segmented" role="tablist" aria-label="Relation type">
          {relationOptions.map((option) => (
            <button
              type="button"
              role="tab"
              aria-selected={relationType === option}
              className={relationType === option ? "active" : ""}
              key={option}
              onClick={() => {
                setRelationType(option);
                setPage(1);
              }}
            >
              {option === "all" ? "All" : relationTypeLabels[option]}
            </button>
          ))}
        </div>
        <form
          className="filter-form"
          onSubmit={(event) => {
            event.preventDefault();
            setPage(1);
            setSubmittedTermId(termId);
          }}
        >
          <label htmlFor="term-id-filter">Term ID</label>
          <input
            id="term-id-filter"
            value={termId}
            onChange={(event) => setTermId(event.target.value)}
            placeholder="term_..."
          />
          <button type="submit">
            <Filter aria-hidden="true" size={16} />
            Apply
          </button>
        </form>
      </div>

      {relations.isLoading ? <LoadingBlock label="Loading relations" /> : null}
      {relations.isError ? <ErrorBlock detail={relations.error.message} /> : null}
      {relations.data && !relations.data.items.length ? (
        <EmptyBlock title="No relations match this filter" />
      ) : null}

      <div className="relation-list">
        {relations.data?.items.map((relation) => (
          <article className="relation-card" key={rawText(relation, ["relation_id"])}>
            <div className="result-title-row">
              <h2>{relationTypeLabels[rawText(relation, ["relation_type"])] || rawText(relation, ["relation_type"])}</h2>
              <ConfidenceBadge value={numberValue(relation.confidence)} />
            </div>
            <p>
              <Link to={`/terms/${encodeURIComponent(rawText(relation, ["source_term_id"]))}`}>
                {rawText(relation, ["source_term"])}
              </Link>
              <span> to </span>
              <Link to={`/terms/${encodeURIComponent(rawText(relation, ["target_term_id"]))}`}>
                {rawText(relation, ["target_term"])}
              </Link>
            </p>
            <div className="chip-list">
              {list(relation.generation_methods).map((method) => (
                <span className="simple-chip" key={String(method)}>
                  {String(method)}
                </span>
              ))}
            </div>
            <pre>{compactJson(relation.evidence)}</pre>
          </article>
        ))}
      </div>
      {relations.data ? <PaginationControls pagination={relations.data.pagination} onPage={setPage} /> : null}
    </section>
  );
}

function list(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function numberValue(value: unknown): number | null {
  return typeof value === "number" ? value : null;
}
