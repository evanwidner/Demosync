import { promises as fs } from "node:fs";
import path from "node:path";
import { randomUUID } from "node:crypto";

const STORAGE_ROOT = process.env.STORAGE_ROOT ?? path.join(process.cwd(), "uploads");

export async function ensureRoot(): Promise<void> {
  await fs.mkdir(STORAGE_ROOT, { recursive: true });
}

export async function saveFile(buffer: Buffer, originalName: string, namespace: string): Promise<{
  storageKey: string;
  absolutePath: string;
  byteSize: number;
}> {
  await ensureRoot();
  const ext = path.extname(originalName).toLowerCase() || ".bin";
  const key = `${namespace}/${randomUUID()}${ext}`;
  const abs = path.join(STORAGE_ROOT, key);
  await fs.mkdir(path.dirname(abs), { recursive: true });
  await fs.writeFile(abs, buffer);
  return { storageKey: key, absolutePath: abs, byteSize: buffer.byteLength };
}

export function publicUrl(storageKey: string): string {
  return `/api/files/${encodeURIComponent(storageKey)}`;
}

export async function readFile(storageKey: string): Promise<{ buffer: Buffer; absolutePath: string }> {
  const abs = path.join(STORAGE_ROOT, storageKey);
  const buffer = await fs.readFile(abs);
  return { buffer, absolutePath: abs };
}
