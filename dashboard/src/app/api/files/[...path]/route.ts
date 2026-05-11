import { NextResponse } from "next/server";
import { readFile } from "@/lib/storage";

const MIME_BY_EXT: Record<string, string> = {
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".webp": "image/webp",
  ".mp4": "video/mp4",
  ".webm": "video/webm",
  ".ply": "application/octet-stream",
  ".json": "application/json",
};

export async function GET(_req: Request, { params }: { params: Promise<{ path: string[] }> }) {
  const { path } = await params;
  const storageKey = path.map((p) => decodeURIComponent(p)).join("/");
  try {
    const { buffer, absolutePath } = await readFile(storageKey);
    const ext = "." + (absolutePath.split(".").pop() ?? "").toLowerCase();
    return new NextResponse(buffer, {
      headers: {
        "Content-Type": MIME_BY_EXT[ext] ?? "application/octet-stream",
        "Cache-Control": "private, max-age=3600",
      },
    });
  } catch {
    return new NextResponse("not found", { status: 404 });
  }
}
