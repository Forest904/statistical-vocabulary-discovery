import { AlertCircle, Loader2, Search } from "lucide-react";
import type { ReactNode } from "react";

export function LoadingBlock({ label = "Loading" }: { label?: string }) {
  return (
    <div className="status-block" role="status">
      <Loader2 aria-hidden="true" className="spin" size={18} />
      <span>{label}</span>
    </div>
  );
}

export function ErrorBlock({ title = "Something went wrong", detail }: { title?: string; detail?: ReactNode }) {
  return (
    <div className="status-block status-error" role="alert">
      <AlertCircle aria-hidden="true" size={18} />
      <div>
        <strong>{title}</strong>
        {detail ? <div className="muted">{detail}</div> : null}
      </div>
    </div>
  );
}

export function EmptyBlock({ title, detail }: { title: string; detail?: string }) {
  return (
    <div className="status-block">
      <Search aria-hidden="true" size={18} />
      <div>
        <strong>{title}</strong>
        {detail ? <div className="muted">{detail}</div> : null}
      </div>
    </div>
  );
}
