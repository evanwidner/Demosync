import { NextResponse } from "next/server";
import { one, query } from "@/lib/db";
import { saveFile } from "@/lib/storage";
import type { Input } from "@/lib/types";

export async function POST(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const property = await one("SELECT id FROM properties WHERE id = $1", [id]);
  if (!property) return NextResponse.json({ error: "property not found" }, { status: 404 });

  const form = await req.formData();
  const files = form.getAll("photos");
  if (files.length === 0) return NextResponse.json({ error: "no photos" }, { status: 400 });

  const inserted: Input[] = [];
  for (const f of files) {
    if (!(f instanceof File)) continue;
    if (!f.type.startsWith("image/")) continue;
    const buf = Buffer.from(await f.arrayBuffer());
    const { storageKey, byteSize } = await saveFile(buf, f.name, `properties/${id}/inputs`);
    const row = await one<Input>(
      `INSERT INTO inputs (property_id, storage_key, original_filename, byte_size, mime_type)
       VALUES ($1, $2, $3, $4, $5) RETURNING *`,
      [id, storageKey, f.name, byteSize, f.type],
    );
    if (row) inserted.push(row);
  }
  return NextResponse.json({ inserted: inserted.length });
}

export async function GET(_req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const rows = await query<Input>("SELECT * FROM inputs WHERE property_id = $1 ORDER BY uploaded_at", [id]);
  return NextResponse.json(rows);
}
