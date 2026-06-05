import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { ConfidenceBadge } from "../components/Badges";
import { PaginationControls } from "../components/Pagination";
import { ErrorBlock, LoadingBlock } from "../components/Status";
import { api } from "../lib/api";
import { formatNumber, rawText } from "../lib/format";
import type { AnyRecord } from "../lib/types";
import { useState } from "react";

export function DomainsPage() {
  const [page, setPage] = useState(1);
  const clusters = useQuery({
    queryKey: ["clusters", page],
    queryFn: () => api.clusters(page, 20)
  });

  if (clusters.isLoading) {
    return <LoadingBlock label="Loading domains" />;
  }
  if (clusters.isError) {
    return <ErrorBlock detail={clusters.error.message} />;
  }

  const grouped = groupByDomain(clusters.data?.items || []);

  return (
    <section className="page-grid">
      <div className="page-heading">
        <p className="eyebrow">Measure-domain browser</p>
        <h1>Inspect clusters by controlled statistical domain.</h1>
      </div>

      {Object.entries(grouped).map(([domain, items]) => (
        <section className="domain-band" key={domain}>
          <div className="section-title">
            <h2>{domain || "Unlabeled domain"}</h2>
            <span>{items.length} clusters on this page</span>
          </div>
          <div className="cluster-grid">
            {items.map((cluster) => (
              <article className="cluster-card" key={rawText(cluster, ["cluster_id"])}>
                <h3>{rawText(cluster, ["cluster_id"])}</h3>
                <p>{formatNumber(cluster.size)} measures</p>
                <h4>Representatives</h4>
                <div className="chip-list">
                  {list(cluster.representatives).map((item) => termLink(item))}
                </div>
                <h4>Members</h4>
                <ul className="plain-list compact">
                  {list(cluster.members)
                    .slice(0, 8)
                    .map((member) => (
                      <li key={rawText(member, ["term_id", "term"])}>
                        {termLink(member)}
                        <ConfidenceBadge value={numberValue(member.membership_probability)} />
                      </li>
                    ))}
                </ul>
              </article>
            ))}
          </div>
        </section>
      ))}
      {clusters.data ? <PaginationControls pagination={clusters.data.pagination} onPage={setPage} /> : null}
    </section>
  );
}

function groupByDomain(items: AnyRecord[]): Record<string, AnyRecord[]> {
  return items.reduce<Record<string, AnyRecord[]>>((groups, item) => {
    const domain = rawText(item, ["domain"]) || "Unlabeled domain";
    groups[domain] = groups[domain] || [];
    groups[domain].push(item);
    return groups;
  }, {});
}

function list(value: unknown): AnyRecord[] {
  return Array.isArray(value) ? (value as AnyRecord[]) : [];
}

function numberValue(value: unknown): number | null {
  return typeof value === "number" ? value : null;
}

function termLink(item: AnyRecord) {
  const termId = rawText(item, ["term_id"]);
  const label = rawText(item, ["term", "canonical_term"]) || termId;
  return termId ? (
    <Link className="simple-chip" to={`/terms/${encodeURIComponent(termId)}`} key={termId}>
      {label}
    </Link>
  ) : (
    <span className="simple-chip" key={label}>
      {label}
    </span>
  );
}
