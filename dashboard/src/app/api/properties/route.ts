import { NextResponse } from "next/server";
import { one } from "@/lib/db";
import type { Property } from "@/lib/types";

export async function POST(req: Request) {
  const body = await req.json().catch(() => null);
  if (!body || typeof body.address !== "string" || !body.address.trim()) {
    return NextResponse.json({ error: "address required" }, { status: 400 });
  }
  const row = await one<Property>(
    `INSERT INTO properties (address, agent_name, listing_description)
     VALUES ($1, $2, $3) RETURNING *`,
    [body.address.trim(), body.agent_name ?? null, body.listing_description ?? null],
  );
  return NextResponse.json(row);
}
