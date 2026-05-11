import { NextResponse } from "next/server";
import { one } from "@/lib/db";
import type { Job } from "@/lib/types";

export async function POST(req: Request) {
  const body = await req.json().catch(() => ({}));
  if (typeof body.property_id !== "string") {
    return NextResponse.json({ error: "property_id required" }, { status: 400 });
  }
  const job = await one<Job>(
    `INSERT INTO jobs (
       property_id, duration_seconds, voiceover_enabled,
       use_claude_organize, use_claude_camera_path, use_claude_qa
     )
     VALUES ($1, $2, $3, $4, $5, $6)
     RETURNING *`,
    [
      body.property_id,
      body.duration_seconds ?? 75,
      body.voiceover_enabled ?? false,
      body.use_claude_organize ?? true,
      body.use_claude_camera_path ?? true,
      body.use_claude_qa ?? true,
    ],
  );
  return NextResponse.json(job);
}
