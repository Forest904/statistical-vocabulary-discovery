import {
  CircleHelp,
  Gauge,
  Layers3,
  MapPinned,
  Ruler,
  Tag,
  Waypoints
} from "lucide-react";

import { confidenceTier, formatPercent, labelForCategory } from "../lib/format";
import type { VocabularyCategory } from "../lib/types";

const categoryIcons = {
  measure: Gauge,
  dimension_name: Layers3,
  dimension_value: Tag,
  unit: Ruler,
  other_ambiguous: CircleHelp
};

export function CategoryBadge({ category }: { category?: string | null }) {
  const safeCategory = category as VocabularyCategory | undefined;
  const Icon = safeCategory && categoryIcons[safeCategory] ? categoryIcons[safeCategory] : Tag;
  return (
    <span className={`badge category category-${category || "unknown"}`}>
      <Icon aria-hidden="true" size={14} />
      {labelForCategory(category)}
    </span>
  );
}

export function ConfidenceBadge({ value }: { value?: number | null }) {
  const tier = confidenceTier(value);
  return (
    <span className={`badge confidence confidence-${tier}`}>
      <Waypoints aria-hidden="true" size={14} />
      {tier === "unknown" ? "Confidence unavailable" : `${tier} confidence ${formatPercent(value)}`}
    </span>
  );
}

export function ConstraintBadge({ kind, label }: { kind: "geo" | "time" | "query"; label: string }) {
  const Icon = kind === "geo" ? MapPinned : kind === "time" ? Gauge : Tag;
  return (
    <span className={`badge constraint constraint-${kind}`}>
      <Icon aria-hidden="true" size={14} />
      {label}
    </span>
  );
}
