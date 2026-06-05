import { sentenceCase } from "../lib/format";

export function ScoreBars({ components }: { components?: Record<string, number> }) {
  const rows = Object.entries(components || {}).filter(([, value]) => Number.isFinite(value));
  if (!rows.length) {
    return <p className="muted">Score components unavailable.</p>;
  }
  return (
    <div className="score-bars" aria-label="Score components">
      {rows.map(([key, value]) => (
        <div className="score-row" key={key}>
          <span>{sentenceCase(key)}</span>
          <div className="meter" aria-hidden="true">
            <span style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
          </div>
          <strong>{value.toFixed(2)}</strong>
        </div>
      ))}
    </div>
  );
}
