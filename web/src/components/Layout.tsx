import { Activity, BookOpen, GitBranch, Network, Search, Tags } from "lucide-react";
import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";

const links = [
  { to: "/", label: "Search", icon: Search },
  { to: "/vocabulary", label: "Vocabulary", icon: Tags },
  { to: "/domains", label: "Domains", icon: Network },
  { to: "/relations", label: "Relations", icon: GitBranch },
  { to: "/methods", label: "Methods", icon: BookOpen }
];

export function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <Activity aria-hidden="true" size={24} />
          <div>
            <strong>StatVocab</strong>
            <span>Grounded table discovery</span>
          </div>
        </div>
        <nav className="main-nav" aria-label="Primary navigation">
          {links.map((link) => {
            const Icon = link.icon;
            return (
              <NavLink key={link.to} to={link.to}>
                <Icon aria-hidden="true" size={17} />
                {link.label}
              </NavLink>
            );
          })}
        </nav>
      </header>
      <main className="content">{children}</main>
    </div>
  );
}
