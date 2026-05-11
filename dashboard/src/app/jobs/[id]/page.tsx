import Link from "next/link";
import { notFound } from "next/navigation";
import { one, query } from "@/lib/db";
import type { Job, Output, Property, QaReport } from "@/lib/types";
import { publicUrl } from "@/lib/storage";
import { JobAutoRefresh } from "./JobAutoRefresh";

export default async function JobPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const job = await one<Job>("SELECT * FROM jobs WHERE id = $1", [id]);
  if (!job) notFound();
  const property = await one<Property>("SELECT * FROM properties WHERE id = $1", [job.property_id]);
  const outputs = await query<Output>("SELECT * FROM outputs WHERE job_id = $1", [job.id]);
  const qa = await one<QaReport>(
    "SELECT * FROM qa_reports WHERE job_id = $1 ORDER BY created_at DESC LIMIT 1",
    [job.id],
  );
  const finalVideo = outputs.find((o) => o.kind === "mp4_16x9") ?? outputs.find((o) => o.kind.startsWith("mp4"));
  const isTerminal = ["complete", "failed", "needs_review"].includes(job.status);

  return (
    <div className="space-y-8">
      {!isTerminal && <JobAutoRefresh />}

      <div className="space-y-1">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-semibold font-mono">{job.id.slice(0, 8)}</h1>
          <span className="text-xs px-2 py-0.5 rounded bg-border">{job.status}</span>
        </div>
        {property && (
          <Link href={`/properties/${property.id}`} className="text-sm text-muted hover:text-accent">
            ← {property.address}
          </Link>
        )}
      </div>

      <StatusTimeline status={job.status} failStage={job.fail_stage} />

      {job.error_message && (
        <div className="border border-bad/40 bg-bad/10 rounded p-4 text-sm whitespace-pre-wrap font-mono">
          {job.error_message}
        </div>
      )}

      {finalVideo && (
        <section className="space-y-2">
          <h2 className="text-lg font-medium">Output</h2>
          <video controls src={publicUrl(finalVideo.storage_key)} className="w-full rounded-lg border border-border bg-black" />
          <div className="text-xs text-muted">
            {outputs.map((o) => (
              <span key={o.id} className="mr-3">{o.kind}</span>
            ))}
          </div>
        </section>
      )}

      {qa && (
        <section className="space-y-2">
          <h2 className="text-lg font-medium">QA report</h2>
          <div className="bg-surface border border-border rounded p-4 space-y-2">
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted">severity:</span>
              <span className={`text-xs px-2 py-0.5 rounded ${
                qa.severity === "ok" ? "bg-emerald-900/40 text-emerald-300" :
                qa.severity === "minor" ? "bg-blue-900/40 text-blue-300" :
                qa.severity === "major" ? "bg-orange-900/40 text-orange-300" :
                "bg-red-900/40 text-red-300"
              }`}>{qa.severity}</span>
            </div>
            {qa.summary && <p className="text-sm">{qa.summary}</p>}
            {Array.isArray(qa.reshoot_requests) && qa.reshoot_requests.length > 0 && (
              <div className="space-y-1 pt-2 border-t border-border">
                <div className="text-sm text-muted">Reshoot recommendations:</div>
                <ul className="text-sm space-y-1">
                  {(qa.reshoot_requests as Array<{ room: string; angle: string; reason: string }>).map((r, i) => (
                    <li key={i} className="ml-4 list-disc">
                      <span className="font-medium">{r.room}:</span> {r.angle} <span className="text-muted">— {r.reason}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </section>
      )}

      <details className="bg-surface border border-border rounded p-4 text-sm">
        <summary className="cursor-pointer text-muted">Job config</summary>
        <pre className="mt-2 text-xs overflow-x-auto">{JSON.stringify(job, null, 2)}</pre>
      </details>
    </div>
  );
}

const STAGES = ["queued", "preprocessing", "colmap", "training", "rendering", "qa", "complete"] as const;

function StatusTimeline({ status, failStage }: { status: string; failStage: string | null }) {
  const failed = status === "failed";
  const currentIndex = failed
    ? STAGES.indexOf(failStage as (typeof STAGES)[number])
    : STAGES.indexOf(status as (typeof STAGES)[number]);
  return (
    <div className="flex items-center gap-2">
      {STAGES.map((s, i) => {
        const reached = currentIndex >= i;
        const active = !failed && i === currentIndex;
        const failedHere = failed && i === currentIndex;
        return (
          <div key={s} className="flex items-center gap-2">
            <div
              className={`px-2 py-1 rounded text-xs ${
                failedHere ? "bg-red-900/40 text-red-300" :
                active ? "bg-blue-900/40 text-blue-300 animate-pulse" :
                reached ? "bg-emerald-900/30 text-emerald-300" :
                "bg-border text-muted"
              }`}
            >
              {s}
            </div>
            {i < STAGES.length - 1 && <span className="text-muted">→</span>}
          </div>
        );
      })}
    </div>
  );
}
