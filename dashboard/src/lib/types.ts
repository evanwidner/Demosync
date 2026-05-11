export type JobStatus =
  | "queued"
  | "preprocessing"
  | "colmap"
  | "training"
  | "rendering"
  | "qa"
  | "complete"
  | "failed"
  | "needs_review";

export interface Property {
  id: string;
  address: string;
  agent_name: string | null;
  listing_description: string | null;
  brokerage_logo_key: string | null;
  created_at: string;
  notes: string | null;
}

export interface Input {
  id: string;
  property_id: string;
  storage_key: string;
  original_filename: string;
  byte_size: number | null;
  mime_type: string | null;
  uploaded_at: string;
  claude_room_label: string | null;
}

export interface Job {
  id: string;
  property_id: string;
  status: JobStatus;
  worker_pod_id: string | null;
  started_at: string | null;
  finished_at: string | null;
  error_message: string | null;
  fail_stage: string | null;
  voiceover_enabled: boolean;
  music_style: string | null;
  duration_seconds: number;
  use_claude_organize: boolean;
  use_claude_camera_path: boolean;
  use_claude_qa: boolean;
  created_at: string;
}

export interface Output {
  id: string;
  job_id: string;
  kind: string;
  storage_key: string;
  public_url: string | null;
  created_at: string;
}

export interface QaReport {
  id: string;
  job_id: string;
  severity: string;
  summary: string | null;
  per_frame: unknown;
  reshoot_requests: unknown;
}
