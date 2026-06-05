import { compactJson, recordEntries, sentenceCase } from "../lib/format";
import type { AnyRecord } from "../lib/types";

export function JsonTable({ record }: { record?: AnyRecord | null }) {
  const rows = recordEntries(record);
  if (!rows.length) {
    return <p className="muted">No details recorded.</p>;
  }
  return (
    <dl className="detail-grid">
      {rows.map(([key, value]) => (
        <div key={key}>
          <dt>{sentenceCase(key)}</dt>
          <dd>{compactJson(value)}</dd>
        </div>
      ))}
    </dl>
  );
}
