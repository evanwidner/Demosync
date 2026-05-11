import Link from "next/link";
import { query } from "@/lib/db";
import type { Property, JobStatus } from "@/lib/types";

interface PropertyRow extends Property {
  job_count: number;
  latest_status: JobStatus | null;
  latest_job_id: string | null;
}

async function loadProperties(): Promise<PropertyRow[]> {
  try {
    return await query<PropertyRow>(
      `SELECT
         p.*,
         COALESCE(j.cnt, 0)::int AS job_count,
         j.latest_status,
         j.latest_job_id
       FROM properties p
       LEFT JOIN LATERAL (
         SELECT COUNT(*) AS cnt,
                (SELECT status FROM jobs WHERE property_id = p.id ORDER BY created_at DESC LIMIT 1) AS latest_status,
                (SELECT id     FROM jobs WHERE property_id = p.id ORDER BY created_at DESC LIMIT 1) AS latest_job_id
         FROM jobs WHERE property_id = p.id
       ) j ON TRUE
       ORDER BY p.created_at DESC`,
    );
  } catch (e) {
    console.error("DB unavailable:", e);
    return [];
  }
}

const statusStyles: Record<string, string> = {
  queued: "bg-border text-muted",
  preprocessing: "bg-yellow-900/40 text-yellow-300",
  colmap: "bg-yellow-900/40 text-yellow-300",
  training: "bg-blue-900/40 text-blue-300",
  rendering: "bg-blue-900/40 text-blue-300",
  qa: "bg-blue-900/40 text-blue-300",
  complete: "bg-emerald-900/40 text-emerald-300",
  failed: "bg-red-900/40 text-red-300",
  needs_review: "bg-orange-900/40 text-orange-300",
};

export default async function Home() {
  const properties = await loadProperties();
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Properties</h1>
        <Link
          href="/properties/new"
          className="text-sm bg-accent text-bg px-4 py-2 rounded font-medium hover:opacity-90"
        >
          + New property
        </Link>
      </div>

      {properties.length === 0 ? (
        <div className="border border-dashed border-border rounded-lg p-12 text-center text-muted">
          <p>No properties yet.</p>
          <p className="mt-1 text-sm">
            <Link href="/properties/new" className="text-accent hover:underline">Create one</Link>{" "}
            and drop in some listing photos to get started.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {properties.map((p) => (
            <Link
              key={p.id}
              href={`/properties/${p.id}`}
              className="block bg-surface border border-border rounded-lg p-4 hover:border-muted transition-colors"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="font-medium truncate">{p.address}</div>
                  {p.agent_name && (
                    <div className="text-xs text-muted mt-0.5 truncate">{p.agent_name}</div>
                  )}
                </div>
                {p.latest_status && (
                  <span className={`text-xs px-2 py-0.5 rounded shrink-0 ${statusStyles[p.latest_status] ?? "bg-border"}`}>
                    {p.latest_status}
                  </span>
                )}
              </div>
              <div className="mt-3 text-xs text-muted flex items-center gap-3">
                <span>{p.job_count} {p.job_count === 1 ? "job" : "jobs"}</span>
                <span>·</span>
                <span>{new Date(p.created_at).toLocaleDateString()}</span>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
