-- DemoSync schema — idempotent, safe to paste into the Supabase SQL editor or
-- run via `psql $DATABASE_URL < schema.sql`.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;  -- pgvector, used by v2 retrieval

CREATE TABLE IF NOT EXISTS properties (
  id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  address             text NOT NULL,
  agent_name          text,
  listing_description text,
  brokerage_logo_key  text,
  created_at          timestamptz NOT NULL DEFAULT now(),
  notes               text
);

CREATE TABLE IF NOT EXISTS inputs (
  id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  property_id        uuid NOT NULL REFERENCES properties(id) ON DELETE CASCADE,
  storage_key        text NOT NULL,
  original_filename  text NOT NULL,
  byte_size          bigint,
  mime_type          text,
  uploaded_at        timestamptz NOT NULL DEFAULT now(),
  claude_room_label  text,
  exif               jsonb
);
CREATE INDEX IF NOT EXISTS inputs_property_idx ON inputs(property_id);

DO $$ BEGIN
  CREATE TYPE job_status AS ENUM (
    'queued', 'preprocessing', 'colmap', 'training', 'rendering', 'qa',
    'complete', 'failed', 'needs_review'
  );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

CREATE TABLE IF NOT EXISTS jobs (
  id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  property_id        uuid NOT NULL REFERENCES properties(id) ON DELETE CASCADE,
  status             job_status NOT NULL DEFAULT 'queued',
  worker_pod_id      text,
  started_at         timestamptz,
  finished_at        timestamptz,
  error_message      text,
  fail_stage         text,
  voiceover_enabled  boolean NOT NULL DEFAULT false,
  music_style        text,
  duration_seconds   integer NOT NULL DEFAULT 75,
  use_claude_organize     boolean NOT NULL DEFAULT true,
  use_claude_camera_path  boolean NOT NULL DEFAULT true,
  use_claude_qa           boolean NOT NULL DEFAULT true,
  created_at         timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS jobs_property_idx ON jobs(property_id);
CREATE INDEX IF NOT EXISTS jobs_status_idx   ON jobs(status);

CREATE TABLE IF NOT EXISTS outputs (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id       uuid NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  kind         text NOT NULL,        -- mp4_16x9 | mp4_9x16 | mp4_1x1 | splat_ply
  storage_key  text NOT NULL,
  public_url   text,
  created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS outputs_job_idx ON outputs(job_id);

CREATE TABLE IF NOT EXISTS qa_reports (
  id                      uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id                  uuid NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  severity                text NOT NULL,
  summary                 text,
  per_frame               jsonb,
  reshoot_requests        jsonb,
  frames_reviewed_keys    text[],
  created_at              timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS qa_reports_job_idx ON qa_reports(job_id);

CREATE TABLE IF NOT EXISTS reconstruction_features (
  id                          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id                      uuid NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  square_footage_estimate     numeric,
  ceiling_height_estimate_ft  numeric,
  room_count                  integer,
  architectural_style_guess   text,
  point_cloud_signature       vector(384),
  created_at                  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS reconstruction_features_signature_idx
  ON reconstruction_features USING ivfflat (point_cloud_signature vector_cosine_ops)
  WITH (lists = 100);

-- Notify daemon when a new job is queued.
CREATE OR REPLACE FUNCTION notify_job_queued() RETURNS trigger AS $$
BEGIN
  IF NEW.status = 'queued' AND (TG_OP = 'INSERT' OR OLD.status IS DISTINCT FROM 'queued') THEN
    PERFORM pg_notify('job_queued', NEW.id::text);
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS jobs_notify_queued ON jobs;
CREATE TRIGGER jobs_notify_queued
  AFTER INSERT OR UPDATE OF status ON jobs
  FOR EACH ROW EXECUTE FUNCTION notify_job_queued();

-- Internal tool: disable RLS so server-side service-role and direct-connection clients
-- can read/write freely. If you ever expose the dashboard publicly, enable RLS and add
-- policies per table.
ALTER TABLE properties              DISABLE ROW LEVEL SECURITY;
ALTER TABLE inputs                  DISABLE ROW LEVEL SECURITY;
ALTER TABLE jobs                    DISABLE ROW LEVEL SECURITY;
ALTER TABLE outputs                 DISABLE ROW LEVEL SECURITY;
ALTER TABLE qa_reports              DISABLE ROW LEVEL SECURITY;
ALTER TABLE reconstruction_features DISABLE ROW LEVEL SECURITY;
