import { useQuery } from "@tanstack/react-query";
import cytoscape from "cytoscape";
import type { Core, ElementDefinition, EventObject } from "cytoscape";
import { ChevronDown, ChevronUp, Focus, RefreshCcw, Search, Settings } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { JsonTable } from "../components/JsonTable";
import { EmptyBlock, ErrorBlock, LoadingBlock } from "../components/Status";
import { api } from "../lib/api";
import { sentenceCase } from "../lib/format";
import type { GraphEdge, GraphFocusOption, GraphNode, GraphNodeType } from "../lib/types";

type FocusSearchType = "all" | Exclude<GraphNodeType, "category">;

const focusTypes: Array<Exclude<GraphNodeType, "category">> = ["term", "table", "cluster", "domain"];
const focusSearchTypes: FocusSearchType[] = ["all", ...focusTypes];
const edgeTypes = [
  "table_contains_term",
  "term_classified_as",
  "measure_member_of_cluster",
  "cluster_belongs_to_domain",
  "broader_than",
  "narrower_than",
  "variant_of",
  "related_to",
  "table_related_to_table"
];

type Selection =
  | { kind: "node"; item: GraphNode }
  | { kind: "edge"; item: GraphEdge }
  | null;

export function GraphPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const focusTypeParam = searchParams.get("focus_type") as Exclude<GraphNodeType, "category"> | null;
  const initialFocusType = focusTypeParam && focusTypes.includes(focusTypeParam)
    ? focusTypeParam
    : "term";
  const [focusType, setFocusType] = useState<FocusSearchType>("all");
  const [searchText, setSearchText] = useState(searchParams.get("focus_id") || "");
  const [selectedSearchText, setSelectedSearchText] = useState("");
  const [submittedSearch, setSubmittedSearch] = useState("");
  const [showResults, setShowResults] = useState(false);
  const [submittedFocus, setSubmittedFocus] = useState({
    focus_type: initialFocusType,
    focus_id: searchParams.get("focus_id") || ""
  });
  const [depth, setDepth] = useState(Number(searchParams.get("depth") || 1));
  const [minWeight, setMinWeight] = useState(Number(searchParams.get("min_weight") || 0));
  const [enabledEdges, setEnabledEdges] = useState<string[]>(edgeTypes);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [selectionOpen, setSelectionOpen] = useState(() =>
    typeof window === "undefined" ? true : !window.matchMedia("(max-width: 820px)").matches
  );
  const [hovered, setHovered] = useState<Selection>(null);
  const [selected, setSelected] = useState<Selection>(null);
  const cyRef = useRef<Core | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const summary = useQuery({
    queryKey: ["graph-summary"],
    queryFn: () => api.graphSummary()
  });

  useEffect(() => {
    const trimmed = searchText.trim();
    if (trimmed.length < 2) {
      setSubmittedSearch("");
      setShowResults(false);
      return undefined;
    }
    if (trimmed === selectedSearchText) {
      setShowResults(false);
      return undefined;
    }
    setShowResults(true);
    const timeout = window.setTimeout(() => {
      setSubmittedSearch(trimmed);
    }, 220);
    return () => window.clearTimeout(timeout);
  }, [searchText]);

  const focusOptions = useQuery({
    queryKey: ["graph-focus-options", submittedSearch, focusType],
    queryFn: () =>
      api.graphFocusOptions({
        q: submittedSearch,
        focus_type: focusType,
        page_size: 10
      }),
    enabled: submittedSearch.trim().length >= 2
  });
  const graph = useQuery({
    queryKey: [
      "graph",
      submittedFocus.focus_type,
      submittedFocus.focus_id,
      depth,
      minWeight,
      enabledEdges
    ],
    queryFn: () =>
      api.graph({
        focus_type: submittedFocus.focus_type,
        focus_id: submittedFocus.focus_id,
        depth,
        min_weight: minWeight,
        edge_type: enabledEdges.length === edgeTypes.length ? undefined : enabledEdges
      }),
    enabled: Boolean(submittedFocus.focus_id)
  });

  useEffect(() => {
    if (focusOptions.data) {
      setShowResults(true);
    }
  }, [focusOptions.data]);

  const nodesById = useMemo(() => {
    const byId = new Map<string, GraphNode>();
    graph.data?.nodes.forEach((node) => byId.set(node.node_id, node));
    return byId;
  }, [graph.data]);

  useEffect(() => {
    if (!containerRef.current || !graph.data) {
      return undefined;
    }
    const nodeLookup = new Map(graph.data.nodes.map((node) => [node.node_id, node]));
    const edgeLookup = new Map(graph.data.edges.map((edge) => [edge.edge_id, edge]));
    const elements: ElementDefinition[] = [
      ...graph.data.nodes.map((node) => ({
        data: {
          id: node.node_id,
          label: node.label,
          nodeType: node.node_type,
          category: String(node.properties.category || "")
        }
      })),
      ...graph.data.edges.map((edge) => ({
        data: {
          id: edge.edge_id,
          source: edge.source_id,
          target: edge.target_id,
          label: edgeLabel(edge.edge_type),
          edgeType: edge.edge_type,
          weight: edge.weight,
          directed: edge.directed ? "true" : "false",
          derived: edge.derived ? "true" : "false"
        }
      }))
    ];
    const cy = cytoscape({
      container: containerRef.current,
      elements,
      layout: { name: "cose", animate: false, padding: 36, nodeRepulsion: 7000 },
      minZoom: 0.22,
      maxZoom: 2.4,
      wheelSensitivity: 0.18,
      style: [
        {
          selector: "node",
          style: {
            "background-color": "#5f7f8d",
            "border-color": "#ffffff",
            "border-width": 2,
            color: "#25332f",
            "font-size": 11,
            height: 28,
            label: "data(label)",
            "min-zoomed-font-size": 8,
            "overlay-opacity": 0,
            "text-background-color": "#ffffff",
            "text-background-opacity": 0.82,
            "text-background-padding": "3px",
            "text-margin-y": -8,
            "text-wrap": "wrap",
            "text-max-width": "120px",
            width: 28
          }
        },
        { selector: 'node[nodeType = "table"]', style: { "background-color": "#3b6c88", shape: "round-rectangle" } },
        { selector: 'node[nodeType = "term"]', style: { "background-color": "#4d8b6f" } },
        { selector: 'node[category = "measure"]', style: { "background-color": "#2f75a0" } },
        { selector: 'node[category = "dimension_name"]', style: { "background-color": "#4f9b6b" } },
        { selector: 'node[category = "dimension_value"]', style: { "background-color": "#b9852b" } },
        { selector: 'node[category = "unit"]', style: { "background-color": "#7566a6" } },
        { selector: 'node[nodeType = "category"]', style: { "background-color": "#7b8794", shape: "tag" } },
        { selector: 'node[nodeType = "cluster"]', style: { "background-color": "#8c6d3f", shape: "hexagon" } },
        { selector: 'node[nodeType = "domain"]', style: { "background-color": "#6f7a3c", shape: "diamond" } },
        {
          selector: "edge",
          style: {
            "curve-style": "bezier",
            color: "#5c6b66",
            "font-size": 9,
            label: "data(label)",
            "line-color": "#9aa8a2",
            opacity: 0.78,
            "target-arrow-color": "#9aa8a2",
            "target-arrow-shape": "triangle",
            width: "mapData(weight, 0, 1, 1, 5)"
          }
        },
        { selector: 'edge[directed = "false"]', style: { "target-arrow-shape": "none" } },
        { selector: 'edge[derived = "true"]', style: { "line-style": "dashed", "line-color": "#7a8b99" } },
        { selector: ".faded", style: { opacity: 0.14, "text-opacity": 0.08 } },
        { selector: ".hovered", style: { "border-color": "#111827", "border-width": 4, opacity: 1 } },
        { selector: "edge.hovered", style: { "line-color": "#111827", "target-arrow-color": "#111827", opacity: 1 } }
      ]
    });
    cyRef.current = cy;

    const clearFocus = () => cy.elements().removeClass("faded hovered");
    const focusElement = (event: EventObject) => {
      const target = event.target;
      cy.elements().addClass("faded");
      target.removeClass("faded").addClass("hovered");
      target.neighborhood().removeClass("faded");
      if (target.isNode()) {
        const node = nodeLookup.get(String(target.id()));
        setHovered(node ? { kind: "node", item: node } : null);
      } else {
        const edge = edgeLookup.get(String(target.id()));
        setHovered(edge ? { kind: "edge", item: edge } : null);
      }
    };
    const blurElement = () => {
      clearFocus();
      setHovered(null);
    };
    const selectElement = (event: EventObject) => {
      const target = event.target;
      if (target.isNode()) {
        const node = nodeLookup.get(String(target.id()));
        setSelected(node ? { kind: "node", item: node } : null);
      } else {
        const edge = edgeLookup.get(String(target.id()));
        setSelected(edge ? { kind: "edge", item: edge } : null);
      }
      setSelectionOpen(true);
    };
    cy.on("mouseover", "node, edge", focusElement);
    cy.on("mouseout", "node, edge", blurElement);
    cy.on("tap", "node, edge", selectElement);
    cy.fit(undefined, 28);

    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, [graph.data]);

  const submitSearch = () => {
    const trimmed = searchText.trim();
    if (trimmed.length < 2) {
      return;
    }
    setSelectedSearchText("");
    setSubmittedSearch(trimmed);
    setShowResults(true);
  };

  const loadFocus = (option: GraphFocusOption) => {
    setSubmittedFocus({ focus_type: option.node_type, focus_id: option.node_id });
    setSearchText(option.label);
    setSelectedSearchText(option.label);
    setSearchParams({
      focus_type: option.node_type,
      focus_id: option.node_id,
      depth: String(depth),
      min_weight: String(minWeight)
    });
    setShowResults(false);
    setSelected(null);
  };

  return (
    <section className="graph-page">
      <div className="page-heading graph-heading">
        <p className="eyebrow">Knowledge graph</p>
        <h1>Explore bounded semantic neighborhoods.</h1>
        {summary.data ? (
          <div className="meta-strip">
            <span>{summary.data.node_count.toLocaleString()} nodes</span>
            <span>{summary.data.edge_count.toLocaleString()} edges</span>
            <span>{summary.data.run_id}</span>
          </div>
        ) : null}
      </div>

      <div className="graph-search-row">
        <form
          className="graph-search-form"
          onSubmit={(event) => {
            event.preventDefault();
            submitSearch();
          }}
        >
          <label htmlFor="graph-search">Graph search</label>
          <div className="graph-search-box">
            <input
              id="graph-search"
              value={searchText}
              onChange={(event) => {
                setSelectedSearchText("");
                setSearchText(event.target.value);
              }}
              placeholder="Search terms, tables, clusters, or domains"
              autoComplete="off"
            />
            <button type="submit" aria-label="Search knowledge graph" disabled={searchText.trim().length < 2}>
              <Search aria-hidden="true" size={17} />
            </button>
          </div>
          {showResults ? (
            <FocusResults
              isLoading={focusOptions.isLoading}
              isError={focusOptions.isError}
              errorMessage={focusOptions.error?.message}
              items={focusOptions.data?.items || []}
              onSelect={loadFocus}
            />
          ) : null}
        </form>
        <button
          className="graph-settings-button"
          type="button"
          aria-expanded={settingsOpen}
          aria-controls="graph-settings-panel"
          onClick={() => setSettingsOpen((current) => !current)}
        >
          <Settings aria-hidden="true" size={17} />
          Settings
          {settingsOpen ? <ChevronUp aria-hidden="true" size={16} /> : <ChevronDown aria-hidden="true" size={16} />}
        </button>
      </div>

      {settingsOpen ? (
        <section className="graph-settings-panel" id="graph-settings-panel">
          <div className="graph-settings-grid">
            <label>
              Search scope
              <select value={focusType} onChange={(event) => setFocusType(event.target.value as FocusSearchType)}>
                {focusSearchTypes.map((type) => (
                  <option key={type} value={type}>
                    {type === "all" ? "All focus nodes" : sentenceCase(type)}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Depth
              <select value={depth} onChange={(event) => setDepth(Number(event.target.value))}>
                <option value={1}>1</option>
                <option value={2}>2</option>
              </select>
            </label>
            <label>
              Weight {minWeight.toFixed(2)}
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={minWeight}
                onChange={(event) => setMinWeight(Number(event.target.value))}
              />
            </label>
            <div className="graph-settings-actions">
              <button type="button" onClick={() => cyRef.current?.fit(undefined, 28)} disabled={!graph.data}>
                <Focus aria-hidden="true" size={16} />
                Fit
              </button>
              <button type="button" onClick={() => graph.refetch()} disabled={!submittedFocus.focus_id}>
                <RefreshCcw aria-hidden="true" size={16} />
                Refresh
              </button>
            </div>
          </div>
          <div className="graph-edge-controls" aria-label="Edge filters">
            {edgeTypes.map((type) => (
              <label className="toggle-chip" key={type}>
                <input
                  type="checkbox"
                  checked={enabledEdges.includes(type)}
                  onChange={(event) => {
                    setEnabledEdges((current) =>
                      event.target.checked ? [...current, type] : current.filter((item) => item !== type)
                    );
                  }}
                />
                {edgeLabel(type)}
              </label>
            ))}
          </div>
        </section>
      ) : null}

      {!submittedFocus.focus_id ? <EmptyBlock title="Choose a graph focus" /> : null}
      {graph.isLoading ? <LoadingBlock label="Loading graph" /> : null}
      {summary.isError ? <ErrorBlock title="Could not load graph summary" detail={summary.error.message} /> : null}
      {graph.isError ? <ErrorBlock title="Could not load graph" detail={graph.error.message} /> : null}

      <div className="graph-workspace">
        <div className="graph-canvas-wrap">
          <div className="graph-canvas" ref={containerRef} aria-label="Knowledge graph canvas" />
          {hovered ? <HoverPreview selection={hovered} nodesById={nodesById} /> : null}
        </div>
        <GraphPanel
          selection={selected}
          nodesById={nodesById}
          isOpen={selectionOpen}
          onToggle={() => setSelectionOpen((current) => !current)}
        />
      </div>
    </section>
  );
}

function FocusResults({
  isLoading,
  isError,
  errorMessage,
  items,
  onSelect
}: {
  isLoading: boolean;
  isError: boolean;
  errorMessage?: string;
  items: GraphFocusOption[];
  onSelect: (option: GraphFocusOption) => void;
}) {
  return (
    <div className="graph-search-results" role="listbox" aria-label="Graph search results">
      {isLoading ? <div className="graph-search-result muted">Searching...</div> : null}
      {isError ? <div className="graph-search-result graph-search-error">{errorMessage || "Search failed"}</div> : null}
      {!isLoading && !isError && !items.length ? (
        <div className="graph-search-result muted">No matching graph nodes.</div>
      ) : null}
      {items.map((item) => (
        <button
          type="button"
          className="graph-search-result"
          key={`${item.node_type}-${item.node_id}`}
          onClick={() => onSelect(item)}
          role="option"
        >
          <span>
            <strong>{item.label}</strong>
            <small>{item.node_id}</small>
          </span>
          <span className="simple-chip">{sentenceCase(item.node_type)}</span>
        </button>
      ))}
    </div>
  );
}

function edgeLabel(type: string): string {
  return sentenceCase(type.replaceAll("_", " "));
}

function HoverPreview({ selection, nodesById }: { selection: Selection; nodesById: Map<string, GraphNode> }) {
  if (!selection) {
    return null;
  }
  const title =
    selection.kind === "node"
      ? selection.item.label
      : `${nodesById.get(selection.item.source_id)?.label || selection.item.source_id} to ${
          nodesById.get(selection.item.target_id)?.label || selection.item.target_id
        }`;
  const subtitle = selection.kind === "node" ? selection.item.node_type : selection.item.edge_type;
  return (
    <div className="graph-hover">
      <strong>{title}</strong>
      <span>{edgeLabel(subtitle)}</span>
    </div>
  );
}

function GraphPanel({
  selection,
  nodesById,
  isOpen,
  onToggle
}: {
  selection: Selection;
  nodesById: Map<string, GraphNode>;
  isOpen: boolean;
  onToggle: () => void;
}) {
  const toggle = (
    <button
      className="graph-panel-toggle"
      type="button"
      aria-expanded={isOpen}
      onClick={onToggle}
    >
      {isOpen ? <ChevronUp aria-hidden="true" size={16} /> : <ChevronDown aria-hidden="true" size={16} />}
    </button>
  );
  if (!selection) {
    return (
      <aside className={`graph-panel ${isOpen ? "" : "collapsed"}`}>
        <div className="graph-panel-header">
          <h2>Selection</h2>
          {toggle}
        </div>
        {isOpen ? <p className="muted">No node or relation selected.</p> : null}
      </aside>
    );
  }
  if (selection.kind === "node") {
    const node = selection.item;
    return (
      <aside className={`graph-panel ${isOpen ? "" : "collapsed"}`}>
        <div className="graph-panel-header">
          <div>
            <h2>{node.label}</h2>
            <span>{sentenceCase(node.node_type)}</span>
          </div>
          {toggle}
        </div>
        {isOpen ? (
          <>
            <div className="chip-list">
              {node.node_type === "term" ? (
                <Link className="simple-chip" to={`/terms/${encodeURIComponent(node.node_id)}`}>
                  Open term
                </Link>
              ) : null}
              {node.node_type === "table" ? (
                <Link className="simple-chip" to={`/tables/${encodeURIComponent(node.node_id)}`}>
                  Open table
                </Link>
              ) : null}
            </div>
            <JsonTable record={{ node_id: node.node_id, ...node.properties }} />
          </>
        ) : null}
      </aside>
    );
  }
  const edge = selection.item;
  const source = nodesById.get(edge.source_id);
  const target = nodesById.get(edge.target_id);
  return (
    <aside className={`graph-panel ${isOpen ? "" : "collapsed"}`}>
      <div className="graph-panel-header">
        <div>
          <h2>{edgeLabel(edge.edge_type)}</h2>
          <span>{edge.weight.toFixed(2)}</span>
        </div>
        {toggle}
      </div>
      {isOpen ? (
        <>
          <p>
            {source?.label || edge.source_id} to {target?.label || edge.target_id}
          </p>
          <div className="chip-list">
            {edge.derived ? <span className="simple-chip">Derived</span> : null}
            {edge.directed ? <span className="simple-chip">Directed</span> : <span className="simple-chip">Undirected</span>}
          </div>
          <JsonTable
            record={{
              edge_id: edge.edge_id,
              evidence_ids: edge.evidence_ids,
              ...edge.properties
            }}
          />
        </>
      ) : null}
    </aside>
  );
}
