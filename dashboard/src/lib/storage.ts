import { promises as fs } from "node:fs";
import path from "node:path";
import { randomUUID } from "node:crypto";

// Legacy local storage root — used only by the /api/files/ fallback route
const STORAGE_ROOT = process.env.STORAGE_ROOT ?? path.join(process.cwd(), "uploads");

export async function saveFile(
  buffer: Buffer,
  originalName: string,
  namespace: string,
): Promise<{ storageKey: string; byteSize: number }> {
  const supabaseUrl = process.env.SUPABASE_URL ?? "";
  const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY ?? "";

  const ext = path.extname(originalName).toLowerCase() || ".bin";
  const key = `${namespace}/${randomUUID()}${ext}`;

  const uploadUrl = `${supabaseUrl}/storage/v1/object/Inputs/${key}`;
  const resp = await fetch(uploadUrl, {
    method: "PUT",
    headers: {
      Authorization: `Bearer ${serviceRoleKey}`,
      "Content-Type": "application/octet-stream",
      "x-upsert": "true",
    },
    body: buffer,
  });

  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`Supabase Storage upload failed: ${resp.status} ${body}`);
  }

  return { storageKey: key, byteSize: buffer.byteLength };
}

export function publicUrl(storageKey: string): string {
  const supabaseUrl = process.env.SUPABASE_URL ?? process.env.NEXT_PUBLIC_SUPABASE_URL ?? "";
  return `${supabaseUrl}/storage/v1/object/public/Inputs/${storageKey}`;
}

// Legacy: serve files written to local disk before Supabase Storage migration
export async function readFile(storageKey: string): Promise<{ buffer: Buffer; absolutePath: string }> {
  const abs = path.join(STORAGE_ROOT, storageKey);
  const buffer = await fs.readFile(abs);
  return { buffer, absolutePath: abs };
}
