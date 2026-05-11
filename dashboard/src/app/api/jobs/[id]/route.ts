import { NextResponse } from "next/server";
import { one, query } from "@/lib/db";
import type { Job, Output, QaReport } from "@/lib/types";

export async function GET(_req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const job = await one<Job>("SELECT * FROM jobs WHERE id = $1", [id]);
  if (!job) return NextResponse.json({ error: "not found" }, { status: 404 });
  const outputs = await query<Output>("SELECT * FROM outputs WHERE job_id = $1", [id]);
  const qa = await one<QaReport>(
    "SELECT * FROM qa_reports WHERE job_id = $1 ORDER BY created_at DESC LIMIT 1",
    [id],
  );
  return NextResponse.json({ job, outputs, qa });
}

export async function PATCH(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const body = await req.json();
  const allowed = [
    "status",
    "worker_pod_id",
    "started_at",
    "finished_at",
    "error_message",
    "fail_stage",
  ];
  const sets: string[] = [];
  const values: unknown[] = [];
  for (const k of allowed) {
    if (k in body) {
      values.push(body[k]);
      sets.push(`${k} = $${values.length}`);
    }
  }
  if (sets.length === 0) return NextResponse.json({ ok: true });
  values.push(id);
  const sql = `UPDATE jobs SET ${sets.join(", ")} WHERE id = $${values.length} RETURNING *`;
  const job = await one<Job>(sql, values);
  return NextResponse.json(job);
}
