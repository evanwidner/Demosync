import { Pool } from "pg";

const globalForPg = globalThis as unknown as { _pg?: Pool };

function buildPool(): Pool {
  const connectionString = process.env.DATABASE_URL;
  if (!connectionString) {
    throw new Error(
      "DATABASE_URL not set. Fill it in dashboard/.env.local (Supabase → Project Settings → Database → Connection string).",
    );
  }
  // Supabase requires SSL. The pg driver picks up sslmode= from the URL, but we set
  // ssl explicitly for cases where users paste the URL without ?sslmode=require.
  const isSupabase = /supabase\.(co|in)/.test(connectionString);
  return new Pool({
    connectionString,
    ssl: isSupabase ? { rejectUnauthorized: false } : undefined,
    max: 5,
  });
}

export const pool: Pool = globalForPg._pg ?? buildPool();
if (process.env.NODE_ENV !== "production") globalForPg._pg = pool;

export async function query<T = unknown>(text: string, params: unknown[] = []): Promise<T[]> {
  const res = await pool.query(text, params);
  return res.rows as T[];
}

export async function one<T = unknown>(text: string, params: unknown[] = []): Promise<T | null> {
  const rows = await query<T>(text, params);
  return rows[0] ?? null;
}
