import { useMutation, useQuery, useQueryClient, type UseMutationResult } from "@tanstack/react-query";
import { Check, ClipboardCheck, RefreshCcw, Send, Wand2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { CategoryBadge } from "../components/Badges";
import { EmptyBlock, ErrorBlock, LoadingBlock } from "../components/Status";
import { api } from "../lib/api";
import { compactJson, formatNumber, labelForCategory, rawText, sentenceCase } from "../lib/format";
import type { AnyRecord, ReviewMode, ReviewTask } from "../lib/types";

const modes: Array<{ value: ReviewMode; label: string }> = [
  { value: "all", label: "All" },
  { value: "term_classification", label: "Classify" },
  { value: "measure_vs_breakdown", label: "Measure or breakdown" },
  { value: "title_reclaim", label: "Title reclaim" },
  { value: "evidence_validation", label: "Evidence" },
  { value: "same_as", label: "Same as" }
];

export function ReviewPage() {
  const queryClient = useQueryClient();
  const [mode, setMode] = useState<ReviewMode>("all");
  const [lastAnswer, setLastAnswer] = useState<string | null>(null);
  const pendingTaskId = useRef<string | null>(null);
  const task = useQuery({
    queryKey: ["review-task", mode],
    queryFn: () => api.reviewTask(mode)
  });
  const stats = useQuery({
    queryKey: ["review-stats"],
    queryFn: () => api.reviewStats()
  });
  const answer = useMutation({
    mutationFn: (value: string) =>
      api.reviewAnswer({
        task_id: task.data?.task_id || "",
        answer: value,
        reviewer_id: "local_user"
      }),
    onSuccess: (_data, value) => {
      setLastAnswer(value);
      queryClient.invalidateQueries({ queryKey: ["review-task"] });
      queryClient.invalidateQueries({ queryKey: ["review-stats"] });
    },
    onSettled: () => {
      pendingTaskId.current = null;
    }
  });
  const compile = useMutation({
    mutationFn: () => api.compileReview(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["review-stats"] });
    }
  });

  const submitAnswer = (value: string) => {
    if (!task.data || pendingTaskId.current === task.data.task_id) {
      return;
    }
    pendingTaskId.current = task.data.task_id;
    answer.mutate(value);
  };

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (!task.data || answer.isPending) {
        return;
      }
      const choice = task.data.choices.find((item) => item.shortcut === event.key);
      if (choice) {
        event.preventDefault();
        submitAnswer(choice.value);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [answer.isPending, task.data]);

  return (
    <section className="page-grid review-page">
      <div className="page-heading">
        <p className="eyebrow">Human loop</p>
        <h1>Review high-impact vocabulary decisions.</h1>
        <p>Answer compact tasks that feed back into gold labels and evaluation metrics.</p>
      </div>

      <div className="toolbar">
        <div className="segmented" role="tablist" aria-label="Review task type">
          {modes.map((item) => (
            <button
              key={item.value}
              type="button"
              role="tab"
              aria-selected={mode === item.value}
              className={mode === item.value ? "active" : ""}
              onClick={() => setMode(item.value)}
            >
              {item.label}
            </button>
          ))}
        </div>
        <button type="button" onClick={() => task.refetch()} disabled={task.isFetching}>
          <RefreshCcw aria-hidden="true" size={16} />
          Refresh
        </button>
      </div>

      <div className="review-layout">
        <div className="review-main">
          {task.isLoading ? <LoadingBlock label="Loading review task" /> : null}
          {task.isError ? <ErrorBlock detail={task.error.message} /> : null}
          {task.data ? (
            <ReviewCard task={task.data} onAnswer={submitAnswer} busy={answer.isPending} />
          ) : null}
          {task.data === null ? (
            <EmptyBlock title="No review task available" detail="Try another mode or generate review tasks." />
          ) : null}
          {answer.isError ? <ErrorBlock detail={answer.error.message} /> : null}
          {lastAnswer ? (
            <p className="compact-note">
              Saved answer: <strong>{sentenceCase(lastAnswer)}</strong>
            </p>
          ) : null}
        </div>

        <aside className="review-side">
          <StatsPanel stats={stats.data} />
          <CompilePanel mutation={compile} />
        </aside>
      </div>
    </section>
  );
}

function ReviewCard({
  task,
  onAnswer,
  busy
}: {
  task: ReviewTask;
  onAnswer: (value: string) => void;
  busy: boolean;
}) {
  const currentCategory = rawText(task.context, ["current_category"]);
  const confidence = rawText(task.context, ["confidence"]);
  return (
    <article className="review-card">
      <div className="review-card-header">
        <span className="badge">
          <ClipboardCheck aria-hidden="true" size={15} />
          {sentenceCase(task.task_type)}
        </span>
        {task.hidden_qc ? <span className="badge">QC repeat</span> : null}
      </div>
      <div className="review-prompt">
        <p>{task.question}</p>
        <h2>{task.canonical_term}</h2>
        {task.paired_canonical_term ? <h3>{task.paired_canonical_term}</h3> : null}
      </div>

      <div className="meta-strip">
        {currentCategory ? <CategoryBadge category={currentCategory} /> : null}
        {currentCategory ? <span>{labelForCategory(currentCategory)}</span> : null}
        {confidence ? <span>Confidence {confidence}</span> : null}
        <span>{task.term_id}</span>
      </div>

      <div className="answer-grid">
        {task.choices.map((choice) => (
          <button
            type="button"
            key={choice.value}
            disabled={busy}
            onClick={() => onAnswer(choice.value)}
          >
            <kbd>{choice.shortcut}</kbd>
            {choice.label}
          </button>
        ))}
      </div>

      <EvidencePanel context={task.context} metadata={task.metadata} />
    </article>
  );
}

function EvidencePanel({ context, metadata }: { context: AnyRecord; metadata: AnyRecord }) {
  const rows = [
    ["Table", rawText(context, ["table_title"])],
    ["Table ID", rawText(context, ["table_id"])],
    ["Evidence", rawText(context, ["evidence"])],
    ["Source area", rawText(context, ["source_area"])],
    ["Metadata column", rawText(context, ["metadata_column"])],
    ["Occurrence", rawText(context, ["occurrence_id"])]
  ].filter(([, value]) => value);

  return (
    <div className="review-evidence">
      <h3>Evidence</h3>
      {rows.length ? (
        <dl>
          {rows.map(([label, value]) => (
            <div key={label}>
              <dt>{label}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
      ) : (
        <p className="muted">No focused evidence available.</p>
      )}
      <details>
        <summary>Task metadata</summary>
        <pre>{compactJson(metadata)}</pre>
      </details>
    </div>
  );
}

function StatsPanel({ stats }: { stats?: Awaited<ReturnType<typeof api.reviewStats>> }) {
  return (
    <div className="panel review-panel">
      <h2>Progress</h2>
      {stats ? (
        <>
          <div className="review-stats-grid">
            <span>
              <strong>{formatNumber(stats.answered_task_count)}</strong>
              answered
            </span>
            <span>
              <strong>{formatNumber(stats.remaining_task_count)}</strong>
              remaining
            </span>
            <span>
              <strong>{formatNumber(stats.event_count)}</strong>
              events
            </span>
          </div>
          <h3>Task mix</h3>
          <ul className="review-count-list">
            {Object.entries(stats.task_type_counts).map(([key, value]) => (
              <li key={key}>
                <span>{sentenceCase(key)}</span>
                <strong>{formatNumber(value)}</strong>
              </li>
            ))}
          </ul>
        </>
      ) : (
        <LoadingBlock label="Loading review stats" />
      )}
    </div>
  );
}

function CompilePanel({
  mutation
}: {
  mutation: UseMutationResult<Awaited<ReturnType<typeof api.compileReview>>, Error, void>;
}) {
  const data = mutation.data;
  return (
    <div className="panel review-panel">
      <h2>Compile</h2>
      <p className="muted">Compile reviewed answers into human-loop labels and refresh classification metrics.</p>
      <button type="button" onClick={() => mutation.mutate()} disabled={mutation.isPending}>
        {mutation.isPending ? <Wand2 aria-hidden="true" size={16} /> : <Send aria-hidden="true" size={16} />}
        Compile labels
      </button>
      {mutation.isError ? <ErrorBlock detail={mutation.error.message} /> : null}
      {data ? (
        <div className="compile-result">
          <p>
            <Check aria-hidden="true" size={16} />
            Compiled {formatNumber(data.compiled_classification_label_count)} labels
          </p>
          <dl>
            <div>
              <dt>Macro F1</dt>
              <dd>{compactJson(data.classification_metrics_summary.macro_f1)}</dd>
            </div>
            <div>
              <dt>Events</dt>
              <dd>{formatNumber(data.events_count)}</dd>
            </div>
          </dl>
        </div>
      ) : null}
    </div>
  );
}
