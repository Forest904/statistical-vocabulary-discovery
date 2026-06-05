import { ChevronLeft, ChevronRight } from "lucide-react";

import type { Pagination } from "../lib/types";

export function PaginationControls({
  pagination,
  onPage
}: {
  pagination: Pagination;
  onPage: (page: number) => void;
}) {
  const maxPage = Math.max(1, Math.ceil(pagination.total / pagination.page_size));
  return (
    <nav className="pagination" aria-label="Pagination">
      <button
        className="icon-button"
        type="button"
        disabled={pagination.page <= 1}
        onClick={() => onPage(pagination.page - 1)}
        aria-label="Previous page"
      >
        <ChevronLeft aria-hidden="true" size={18} />
      </button>
      <span>
        Page {pagination.page} of {maxPage} - {pagination.total.toLocaleString()} items
      </span>
      <button
        className="icon-button"
        type="button"
        disabled={pagination.page >= maxPage}
        onClick={() => onPage(pagination.page + 1)}
        aria-label="Next page"
      >
        <ChevronRight aria-hidden="true" size={18} />
      </button>
    </nav>
  );
}
