import { ChevronDown } from "lucide-react";
import { useState, type ReactNode } from "react";

import { compactJson } from "../lib/format";

export function EvidenceDisclosure({
  title = "Raw evidence",
  summary,
  value,
  children
}: {
  title?: string;
  summary?: ReactNode;
  value?: unknown;
  children?: ReactNode;
}) {
  const [open, setOpen] = useState(false);

  return (
    <details className="evidence-disclosure" onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>
        <span>
          <strong>{title}</strong>
          {summary ? <span className="muted">{summary}</span> : null}
        </span>
        <ChevronDown aria-hidden="true" size={16} />
      </summary>
      {open ? children || <pre>{compactJson(value)}</pre> : null}
    </details>
  );
}

export function EvidenceFacts({ record }: { record?: Record<string, unknown> | null }) {
  const facts = Object.entries(record || {})
    .filter(([, value]) => value !== null && value !== undefined && value !== "")
    .slice(0, 4);

  if (!facts.length) {
    return null;
  }

  return (
    <dl className="fact-strip">
      {facts.map(([key, value]) => (
        <div key={key}>
          <dt>{key.replaceAll("_", " ")}</dt>
          <dd>{compactJson(value)}</dd>
        </div>
      ))}
    </dl>
  );
}
