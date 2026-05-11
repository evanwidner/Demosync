import Link from "next/link";
import { notFound } from "next/navigation";
import { query, one } from "@/lib/db";
import type { Property, Input, Job, Output } from "@/lib/types";
import { publicUrl } from "@/lib/storage";
import { CreateJobButton } from "./CreateJobButton";

export default async function PropertyPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const property = await one<Property>("SELECT * FROM properties WHERE id = $1", [id]);
  if (!property) notFound();
  const inputs = await query<Input>(
    "SELECT * FROM inputs WHERE property_id = $1 ORDER BY uploaded_at ASC",
    [id],
  );
  const jobs = await query<Job>(
    "SELECT * FROM jobs WHERE property_id = $1 ORDER BY created_at DESC",
    [id],
  );
  const latestJob = jobs[0];
  const outputs = latestJob
    ? await query<Output>("SELECT * FROM outputs WHERE job_id = $1", [latestJob.id])
    : [];
  const finalVideo = outputs.find((o) => o.kind === "mp4_16x9") ?? outputs.find((o) => o.kind.startsWith("mp4"));

  return (
    <div className="space-y-8">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">{property.address}</h1>
          {property.agent_name && <div className="text-sm text-muted mt-1">{property.agent_name}</div>}
        </div>
        <CreateJobButton propertyId={property.id} canCreate={inputs.length >= 20} />
      </div>

      {property.listing_description && (
        <details className="bg-surface border border-border rounded p-4">
          <summary className="cursor-pointer text-sm text-muted">Listing description</summary>
          <p className="mt-2 text-sm whitespace-pre-wrap">{property.listing_description}</p>
        </details>
      )}

      {finalVideo && (
        <section className="space-y-2">
          <h2 className="text-lg font-medium">Latest render</h2>
          <video
            controls
            src={finalVideo.public_url ?? ""}
            className="w-full rounded-lg border border-border bg-black"
          />
        </section>
      )}

      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-medium">Photos</h2>
          <span className="text-sm text-muted">{inputs.length}</span>
        </div>
        {inputs.length === 0 ? (
          <p className="text-sm text-muted">No photos uploaded yet.</p>
        ) : (
          <div className="grid grid-cols-6 sm:grid-cols-8 gap-2">
            {inputs.map((p) => (
              <div key={p.id} className="aspect-square bg-surface border border-border rounded overflow-hidden">
                <img src={publicUrl(p.storage_key)} alt={p.original_filename} className="w-full h-full object-cover" />
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-medium">Jobs</h2>
        {jobs.length === 0 ? (
          <p className="text-sm text-muted">No jobs run yet.</p>
        ) : (
          <ul className="divide-y divide-border bg-surface border border-border rounded">
            {jobs.map((j) => (
              <li key={j.id} className="px-4 py-3 flex items-center justify-between">
                <div>
                  <Link href={`/jobs/${j.id}`} className="font-mono text-sm hover:text-accent">
                    {j.id.slice(0, 8)}
                  </Link>
                  <div className="text-xs text-muted mt-0.5">
                    {new Date(j.created_at).toLocaleString()} · {j.duration_seconds}s
                  </div>
                </div>
                <span className="text-xs px-2 py-0.5 rounded bg-border">{j.status}</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
