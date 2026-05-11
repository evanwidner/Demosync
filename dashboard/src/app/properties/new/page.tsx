"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useDropzone } from "react-dropzone";

export default function NewPropertyPage() {
  const router = useRouter();
  const [address, setAddress] = useState("");
  const [agentName, setAgentName] = useState("");
  const [description, setDescription] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { "image/jpeg": [".jpg", ".jpeg"], "image/png": [".png"], "image/webp": [".webp"] },
    onDrop: (accepted) => setFiles((prev) => [...prev, ...accepted]),
  });

  async function handleSubmit() {
    if (!address.trim()) {
      setError("Address is required");
      return;
    }
    if (files.length < 20) {
      setError(`Need at least 20 photos (have ${files.length}). 30-60 works best.`);
      return;
    }
    setError(null);
    setSubmitting(true);
    try {
      const resp = await fetch("/api/properties", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          address: address.trim(),
          agent_name: agentName.trim() || null,
          listing_description: description.trim() || null,
        }),
      });
      if (!resp.ok) throw new Error(`create property failed (${resp.status})`);
      const { id } = await resp.json();

      const form = new FormData();
      for (const f of files) form.append("photos", f, f.name);
      const upload = await fetch(`/api/properties/${id}/photos`, { method: "POST", body: form });
      if (!upload.ok) throw new Error(`photo upload failed (${upload.status})`);

      router.push(`/properties/${id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
      setSubmitting(false);
    }
  }

  return (
    <div className="space-y-8 max-w-3xl">
      <h1 className="text-2xl font-semibold">New property</h1>

      <div className="space-y-4">
        <Field label="Address">
          <input
            value={address}
            onChange={(e) => setAddress(e.target.value)}
            placeholder="123 Main St, Corrales NM"
            className="w-full bg-surface border border-border rounded px-3 py-2"
          />
        </Field>

        <Field label="Agent name (optional)">
          <input
            value={agentName}
            onChange={(e) => setAgentName(e.target.value)}
            className="w-full bg-surface border border-border rounded px-3 py-2"
          />
        </Field>

        <Field label="Listing description (optional, used by Claude for hero-feature targeting)">
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            rows={4}
            className="w-full bg-surface border border-border rounded px-3 py-2 font-mono text-sm"
            placeholder="Stunning Spanish Mediterranean with Sandia views, vaulted ceilings, gourmet kitchen..."
          />
        </Field>
      </div>

      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-10 text-center cursor-pointer transition-colors ${
          isDragActive ? "border-accent bg-accent/5" : "border-border hover:border-muted"
        }`}
      >
        <input {...getInputProps()} />
        <p className="text-lg">Drop listing photos here</p>
        <p className="text-sm text-muted mt-1">
          {files.length === 0
            ? "JPG / PNG / WebP. Need 20-69. 30-60 recommended."
            : `${files.length} photo${files.length === 1 ? "" : "s"} staged`}
        </p>
      </div>

      {files.length > 0 && (
        <div className="grid grid-cols-6 sm:grid-cols-8 gap-2">
          {files.slice(0, 32).map((f, i) => (
            <div key={i} className="aspect-square bg-surface border border-border rounded overflow-hidden">
              <img src={URL.createObjectURL(f)} alt="" className="w-full h-full object-cover" />
            </div>
          ))}
          {files.length > 32 && (
            <div className="aspect-square bg-surface border border-border rounded flex items-center justify-center text-xs text-muted">
              +{files.length - 32}
            </div>
          )}
        </div>
      )}

      {error && <div className="text-bad text-sm">{error}</div>}

      <div className="flex gap-3">
        <button
          onClick={handleSubmit}
          disabled={submitting || files.length === 0 || !address.trim()}
          className="bg-accent text-bg px-5 py-2 rounded font-medium disabled:opacity-40"
        >
          {submitting ? "Uploading…" : "Create property"}
        </button>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <div className="text-sm text-muted mb-1">{label}</div>
      {children}
    </label>
  );
}
