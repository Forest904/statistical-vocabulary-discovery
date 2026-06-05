import { useMutation } from "@tanstack/react-query";
import { ExternalLink, FileText, Search } from "lucide-react";
import { FormEvent, useState } from "react";
import { Link } from "react-router-dom";

import { CategoryBadge, ConstraintBadge } from "../components/Badges";
import { ScoreBars } from "../components/ScoreBars";
import { EmptyBlock, ErrorBlock, LoadingBlock } from "../components/Status";
import { api } from "../lib/api";
import { compactJson, rawText } from "../lib/format";
import type { SearchResult, SearchSystem } from "../lib/types";

const systems: Array<{ value: SearchSystem; label: string }> = [
  { value: "fused", label: "Fused" },
  { value: "title_bm25", label: "Title BM25" },
  { value: "all_vocabulary_bm25", label: "Vocabulary BM25" },
  { value: "pearl_semantic", label: "PEARL semantic" }
];

export function SearchPage() {
  const [query, setQuery] = useState("employment in Italy 2020");
  const [system, setSystem] = useState<SearchSystem>("fused");
  const [selected, setSelected] = useState<SearchResult | null>(null);
  const canSearch = query.trim().length > 0;
  const search = useMutation({
    mutationFn: (input: { query: string; system: SearchSystem }) =>
      api.search({ ...input, page: 1, page_size: 10 })
  });

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmedQuery = query.trim();
    if (!trimmedQuery) {
      search.reset();
      return;
    }
    setSelected(null);
    search.mutate({ query: trimmedQuery, system });
  };

  return (
    <section className="page-grid">
      <div className="page-heading">
        <p className="eyebrow">Journalism search</p>
        <h1>Find source tables with visible evidence.</h1>
        <p>
          StatVocab finds relevant Eurostat-style source tables; it does not return a numeric
          answer.
        </p>
      </div>

      <form className="search-panel" onSubmit={submit}>
        <label htmlFor="query">Search query</label>
        <div className="search-row">
          <input
            id="query"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Example: unemployment rate in Italy 2020"
            required
          />
          <select
            aria-label="Retrieval system"
            value={system}
            onChange={(event) => setSystem(event.target.value as SearchSystem)}
          >
            {systems.map((item) => (
              <option key={item.value} value={item.value}>
                {item.label}
              </option>
            ))}
          </select>
          <button type="submit" disabled={!canSearch}>
            <Search aria-hidden="true" size={17} />
            Search
          </button>
        </div>
      </form>

      {search.isPending ? <LoadingBlock label="Searching tables" /> : null}
      {search.isError ? <ErrorBlock detail={search.error.message} /> : null}

      {search.data ? (
        <div className="results-layout">
          <aside className="parsed-panel" aria-label="Parsed query">
            <h2>Parsed Query</h2>
            <div className="chip-list">
              <ConstraintBadge
                kind="query"
                label={`Semantic: ${search.data.query.semantic_remainder || "none"}`}
              />
              {(search.data.query.geographies || []).map((item, index) => (
                <ConstraintBadge
                  key={`${rawText(item, ["code", "name"])}-${index}`}
                  kind="geo"
                  label={rawText(item, ["name", "code", "raw_value"]) || "Geography"}
                />
              ))}
              {(search.data.query.times || []).map((item, index) => (
                <ConstraintBadge
                  key={`${rawText(item, ["normalized_value", "raw_value"])}-${index}`}
                  kind="time"
                  label={rawText(item, ["normalized_value", "raw_value"]) || "Time"}
                />
              ))}
            </div>
            <p className="notice">{search.data.notice}</p>
          </aside>

          <div className="result-stack">
            {search.data.results.length ? (
              search.data.results.map((result) => (
                <article className="result-card" key={result.table_id}>
                  <div className="result-rank">#{result.rank}</div>
                  <div className="result-body">
                    <div className="result-title-row">
                      <h2>{result.title || result.table_id}</h2>
                      <strong>{result.score.toFixed(2)}</strong>
                    </div>
                    <p className="muted">{result.table_id}</p>
                    <div className="chip-list">
                      {(result.matched_terms || []).slice(0, 8).map((term) => (
                        <span className="term-chip" key={`${term.category}-${term.term}`}>
                          <CategoryBadge category={term.category} />
                          {term.term}
                        </span>
                      ))}
                    </div>
                    <ScoreBars components={result.score_components} />
                    <div className="card-actions">
                      <button type="button" onClick={() => setSelected(result)}>
                        <FileText aria-hidden="true" size={16} />
                        Explain
                      </button>
                      <Link to={`/tables/${encodeURIComponent(result.table_id)}`}>
                        Open table
                      </Link>
                      {result.source_url ? (
                        <a href={result.source_url} target="_blank" rel="noreferrer">
                          <ExternalLink aria-hidden="true" size={16} />
                          Source
                        </a>
                      ) : null}
                    </div>
                  </div>
                </article>
              ))
            ) : (
              <EmptyBlock
                title="No ranked tables found"
                detail="Try a broader measure, geography, or time expression."
              />
            )}
          </div>
        </div>
      ) : (
        <EmptyBlock title="Ready to search" detail="Submit a query to inspect grounded tables." />
      )}

      {selected ? <ExplanationDrawer result={selected} onClose={() => setSelected(null)} /> : null}
    </section>
  );
}

function ExplanationDrawer({ result, onClose }: { result: SearchResult; onClose: () => void }) {
  const warnings = Array.isArray(result.evidence?.warnings) ? result.evidence.warnings : [];
  return (
    <div className="drawer-backdrop" role="presentation">
      <aside className="drawer" aria-label="Result explanation">
        <div className="drawer-header">
          <div>
            <p className="eyebrow">Evidence for rank {result.rank}</p>
            <h2>{result.title}</h2>
          </div>
          <button
            autoFocus
            className="icon-button"
            type="button"
            onClick={onClose}
            aria-label="Close drawer"
          >
            X
          </button>
        </div>
        <ScoreBars components={result.score_components} />
        <h3>Matched Terms</h3>
        <div className="chip-list">
          {(result.matched_terms || []).map((term) => (
            <span className="term-chip" key={`${term.category}-${term.term}`}>
              <CategoryBadge category={term.category} />
              {term.term}
            </span>
          ))}
        </div>
        <h3>Warnings</h3>
        {warnings.length ? (
          <ul className="plain-list">
            {warnings.map((warning) => (
              <li key={String(warning)}>{String(warning)}</li>
            ))}
          </ul>
        ) : (
          <p className="muted">No warning flags for this result.</p>
        )}
        <h3>Raw Evidence</h3>
        <pre>{compactJson(result.evidence)}</pre>
        <p className="notice">{result.notice}</p>
      </aside>
    </div>
  );
}
