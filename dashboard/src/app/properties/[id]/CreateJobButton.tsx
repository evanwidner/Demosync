"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export function CreateJobButton({ propertyId, canCreate }: { propertyId: string; canCreate: boolean }) {
  const router = useRouter();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handle() {
    setSubmitting(true);
    setError(null);
    try {
      const resp = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ property_id: propertyId }),
      });
      if (!resp.ok) throw new Error(`create job failed (${resp.status})`);
      const { id } = await resp.json();
      router.push(`/jobs/${id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
      setSubmitting(false);
    }
  }

  return (
    <div className="flex flex-col items-end gap-1">
      <button
        onClick={handle}
        disabled={!canCreate || submitting}
        className="bg-accent text-bg px-4 py-2 rounded font-medium text-sm disabled:opacity-40"
        title={canCreate ? "" : "Upload at least 20 photos first"}
      >
        {submitting ? "Queuing…" : "Render walkthrough"}
      </button>
      {error && <div className="text-bad text-xs">{error}</div>}
      {!canCreate && <div className="text-xs text-muted">Need ≥20 photos</div>}
    </div>
  );
}
