import { expect, Page, test } from "@playwright/test";

const notice = "StatVocab finds relevant source tables; it does not return a numeric answer.";

async function mockApi(page: Page) {
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    const method = route.request().method();

    if (path === "/api/search" && method === "POST") {
      await route.fulfill({
        json: {
          query: {
            original: "employment in Italy 2020",
            semantic_remainder: "employment",
            geographies: [{ code: "IT", name: "Italy", raw_value: "Italy" }],
            times: [{ raw_value: "2020", normalized_value: "2020" }]
          },
          notice,
          pagination: { page: 1, page_size: 10, total: 1 },
          results: [
            {
              rank: 1,
              table_id: "table_1",
              title: "Employment by sex, age and citizenship",
              score: 0.92,
              score_components: {
                lexical: 0.8,
                semantic: 0.7,
                measure: 1,
                geography: 1,
                time: 1
              },
              matched_terms: [{ category: "measure", term: "Employment" }],
              source_url: "https://example.test/table",
              evidence: {
                title: "Employment by sex, age and citizenship",
                warnings: ["no unsupported numeric answer returned"]
              },
              notice
            }
          ]
        }
      });
      return;
    }

    if (path === "/api/tables/table_1") {
      await route.fulfill({
        json: {
          table_id: "table_1",
          title: "Employment by sex, age and citizenship",
          source_url: "https://example.test/table",
          parse_status: "parsed",
          metadata: { row_count: 10, observation_count: 80 },
          vocabulary: {
            measures: ["Employment"],
            dimension_names: ["Sex"],
            dimension_values: ["Italy"],
            units: ["Percentage"],
            other_ambiguous: [],
            domains: ["labour market"]
          },
          geographies: [{ code: "IT", name: "Italy" }],
          times: [{ normalized_value: "2020" }],
          evidence: [{ kind: "title", raw_value: "Employment by sex, age and citizenship" }],
          related_terms: [{ term_id: "term_1", canonical_term: "Employment" }]
        }
      });
      return;
    }

    if (path === "/api/terms/term_1") {
      await route.fulfill({
        json: {
          term_id: "term_1",
          canonical_term: "Employment",
          category: "measure",
          confidence: 0.91,
          provenance: {
            table_count: 12,
            occurrence_count: 20,
            evidence: "title and metadata evidence",
            run_id: "run_classification"
          },
          occurrence_ids: ["occ_1"],
          table_appearances: [{ table_id: "table_1" }],
          cluster: {
            cluster_id: "cluster_1",
            domain: "labour market",
            membership_probability: 0.95
          },
          relations: {
            incoming: [],
            outgoing: [
              {
                relation_id: "relation_1",
                source_term_id: "term_1",
                source_term: "Employment",
                target_term_id: "term_2",
                target_term: "Employment rate",
                relation_type: "related_to",
                confidence: 0.88,
                evidence: { embedding_similarity: 0.94 }
              }
            ]
          }
        }
      });
      return;
    }

    if (path === "/api/terms") {
      await route.fulfill({
        json: {
          pagination: { page: Number(url.searchParams.get("page") || 1), page_size: 20, total: 1 },
          items: [
            {
              term_id: "term_1",
              canonical_term: "Employment",
              category: "measure",
              confidence: 0.91,
              table_count: 12,
              occurrence_count: 20,
              cluster: { cluster_id: "cluster_1", domain: "labour market" },
              relation_counts: { incoming: 0, outgoing: 1, total: 1 }
            }
          ]
        }
      });
      return;
    }

    if (path === "/api/clusters") {
      await route.fulfill({
        json: {
          pagination: { page: 1, page_size: 20, total: 1 },
          items: [
            {
              cluster_id: "cluster_1",
              domain: "labour market",
              size: 2,
              representatives: [{ term_id: "term_1", term: "Employment" }],
              members: [
                {
                  term_id: "term_1",
                  term: "Employment",
                  membership_probability: 0.95,
                  is_representative: true
                }
              ]
            }
          ]
        }
      });
      return;
    }

    if (path === "/api/relations") {
      await route.fulfill({
        json: {
          pagination: { page: 1, page_size: 20, total: 1 },
          items: [
            {
              relation_id: "relation_1",
              source_term_id: "term_1",
              source_term: "Employment",
              target_term_id: "term_2",
              target_term: "Employment rate",
              relation_type: "related_to",
              confidence: 0.88,
              generation_methods: ["embedding_similarity", "domains"],
              evidence: { embedding_similarity: 0.94, domain_signal: "same_domain" }
            }
          ]
        }
      });
      return;
    }

    if (path === "/api/health") {
      await route.fulfill({
        json: {
          status: "ok",
          config_name: "core",
          corpus: "core",
          loaded_search_run_id: "run_search",
          document_count: 2000,
          readiness: {
            config_loaded: true,
            search_index_ready: true,
            tables_loaded: true,
            terms_loaded: true,
            clusters_loaded: true,
            relations_loaded: true,
            evaluation_loaded: true
          },
          artifact_counts: {
            tables: 2000,
            search_documents: 2000,
            terms: 100,
            clusters: 10,
            relations: 25
          },
          warnings: []
        }
      });
      return;
    }

    if (path === "/api/evaluation") {
      await route.fulfill({
        json: {
          summaries: {
            retrieval: {
              status: "available",
              path: "report/retrieval_metrics.json",
              data: { systems: { fused: { original: { "HitRate@10": 0.95 } } } }
            },
            classification: {
              status: "available",
              path: "report/classification_metrics.json",
              data: { macro_f1: 0.82 }
            }
          }
        }
      });
      return;
    }

    await route.fulfill({ status: 404, json: { error: { code: "not_mocked", message: path } } });
  });
}

test.beforeEach(async ({ page }) => {
  await mockApi(page);
});

test("search renders parsed query, evidence drawer, and table navigation", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Search query").fill("employment in Italy 2020");
  await page.getByRole("button", { name: "Search" }).click();

  await expect(page.getByText("Semantic: employment")).toBeVisible();
  await expect(page.getByText("Employment by sex, age and citizenship")).toBeVisible();
  await expect(page.getByText(notice)).toBeVisible();

  await page.getByRole("button", { name: "Explain" }).click();
  await expect(page.getByLabel("Result explanation")).toContainText("Warnings");
  await expect(page.getByRole("button", { name: "Close drawer" })).toBeFocused();
  await page.getByRole("button", { name: "Close drawer" }).click();

  await page.getByRole("link", { name: "Open table" }).click();
  await expect(page.getByRole("heading", { name: "Employment by sex, age and citizenship" })).toBeVisible();
  await expect(page.getByText("row_count")).not.toBeVisible();
  await expect(page.getByText("Row count")).toBeVisible();
});

test("search does not submit when the query is empty", async ({ page }) => {
  let searchRequests = 0;
  page.on("request", (request) => {
    if (new URL(request.url()).pathname === "/api/search") {
      searchRequests += 1;
    }
  });

  await page.goto("/");
  await page.getByLabel("Search query").fill("   ");
  await expect(page.getByRole("button", { name: "Search" })).toBeDisabled();
  await page.getByLabel("Search query").press("Enter");
  await page.waitForTimeout(100);

  expect(searchRequests).toBe(0);
  await expect(page.getByText("Ready to search")).toBeVisible();
});

test("vocabulary atlas filters and opens term detail", async ({ page }) => {
  await page.goto("/vocabulary");
  await page.getByRole("tab", { name: "Measure" }).click();
  await page.getByLabel("Filter terms").fill("employment");
  await page.getByRole("button", { name: "Filter" }).click();

  await expect(page.getByText("Employment")).toBeVisible();
  await expect(page.getByText("high confidence 91%")).toBeVisible();
  await page.getByRole("link", { name: /Employment/ }).first().click();
  await expect(page.getByRole("heading", { name: "Employment" })).toBeVisible();
  await expect(page.getByText("title and metadata evidence")).toBeVisible();
});

test("domains and relations render browsable evidence", async ({ page }) => {
  await page.goto("/domains");
  await expect(page.getByRole("heading", { name: "labour market" })).toBeVisible();
  await expect(page.getByText("cluster_1")).toBeVisible();

  await page.goto("/relations");
  await page.getByRole("tab", { name: "Related to" }).click();
  await expect(page.getByText("Employment rate")).toBeVisible();
  await expect(page.getByText("embedding_similarity", { exact: true })).toBeVisible();
});

test("methods page states scope and readiness", async ({ page }) => {
  await page.goto("/methods");
  await expect(page.getByText("never unsupported numeric answers")).toBeVisible();
  await expect(page.getByText("2,000 indexed documents")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Retrieval" })).toBeVisible();
  await expect(page.getByText("HitRate@10")).toBeVisible();
});
