import { Route, Routes } from "react-router-dom";

import { Layout } from "./components/Layout";
import { DomainsPage } from "./pages/DomainsPage";
import { MethodsPage } from "./pages/MethodsPage";
import { RelationsPage } from "./pages/RelationsPage";
import { SearchPage } from "./pages/SearchPage";
import { TableDetailPage } from "./pages/TableDetailPage";
import { TermDetailPage } from "./pages/TermDetailPage";
import { VocabularyPage } from "./pages/VocabularyPage";

export function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<SearchPage />} />
        <Route path="/tables/:tableId" element={<TableDetailPage />} />
        <Route path="/vocabulary" element={<VocabularyPage />} />
        <Route path="/terms/:termId" element={<TermDetailPage />} />
        <Route path="/domains" element={<DomainsPage />} />
        <Route path="/relations" element={<RelationsPage />} />
        <Route path="/methods" element={<MethodsPage />} />
      </Routes>
    </Layout>
  );
}
