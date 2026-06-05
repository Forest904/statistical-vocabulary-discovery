import { useQuery } from "@tanstack/react-query";
import { Search } from "lucide-react";
import { FormEvent, useState } from "react";
import { Link } from "react-router-dom";

import { CategoryBadge, ConfidenceBadge } from "../components/Badges";
import { PaginationControls } from "../components/Pagination";
import { EmptyBlock, ErrorBlock, LoadingBlock } from "../components/Status";
import { api } from "../lib/api";
import { categoryOptions, formatNumber, labelForCategory } from "../lib/format";
import type { VocabularyCategory } from "../lib/types";

export function VocabularyPage() {
  const [category, setCategory] = useState<VocabularyCategory | "all">("all");
  const [text, setText] = useState("");
  const [submittedText, setSubmittedText] = useState("");
  const [page, setPage] = useState(1);
  const terms = useQuery({
    queryKey: ["terms", category, submittedText, page],
    queryFn: () => api.terms({ category, q: submittedText, page, page_size: 20 })
  });

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setPage(1);
    setSubmittedText(text);
  };

  return (
    <section className="page-grid">
      <div className="page-heading">
        <p className="eyebrow">Vocabulary atlas</p>
        <h1>Browse extracted terms and their provenance.</h1>
      </div>

      <div className="toolbar">
        <div className="segmented" role="tablist" aria-label="Vocabulary category">
          {categoryOptions.map((option) => (
            <button
              type="button"
              role="tab"
              aria-selected={category === option}
              className={category === option ? "active" : ""}
              key={option}
              onClick={() => {
                setCategory(option);
                setPage(1);
              }}
            >
              {option === "all" ? "All" : labelForCategory(option)}
            </button>
          ))}
        </div>
        <form className="filter-form" onSubmit={submit}>
          <label htmlFor="term-filter">Filter terms</label>
          <input
            id="term-filter"
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="term text"
          />
          <button type="submit">
            <Search aria-hidden="true" size={16} />
            Filter
          </button>
        </form>
      </div>

      {terms.isLoading ? <LoadingBlock label="Loading vocabulary" /> : null}
      {terms.isError ? <ErrorBlock detail={terms.error.message} /> : null}
      {terms.data && !terms.data.items.length ? (
        <EmptyBlock title="No terms match this view" detail="Try another category or filter." />
      ) : null}

      {terms.data?.items.length ? (
        <>
          <div className="table-list">
            {terms.data.items.map((term) => (
              <Link className="term-row" to={`/terms/${encodeURIComponent(term.term_id)}`} key={term.term_id}>
                <div>
                  <strong>{term.canonical_term}</strong>
                  <span className="record-id">{term.term_id}</span>
                </div>
                <CategoryBadge category={term.category} />
                <ConfidenceBadge value={term.confidence} />
                <span className="stat-cell">
                  <strong>{formatNumber(term.table_count)}</strong>
                  tables
                </span>
                <span className="stat-cell">
                  <strong>{formatNumber(term.occurrence_count)}</strong>
                  occurrences
                </span>
                <span className="stat-cell">
                  <strong>{formatNumber(term.relation_counts?.total || 0)}</strong>
                  relations
                </span>
              </Link>
            ))}
          </div>
          <PaginationControls pagination={terms.data.pagination} onPage={setPage} />
        </>
      ) : null}
    </section>
  );
}
