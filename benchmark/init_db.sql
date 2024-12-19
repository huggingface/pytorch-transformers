CREATE TABLE IF NOT EXISTS benchmarks (
  benchmark_id SERIAL PRIMARY KEY,
  branch VARCHAR(255),
  commit_id VARCHAR(72),
  commit_message VARCHAR(70),
  metadata jsonb,
  created_at timestamp without time zone NOT NULL DEFAULT (current_timestamp AT TIME ZONE 'UTC')
);

CREATE INDEX IF NOT EXISTS benchmarks_benchmark_id_idx ON benchmarks (benchmark_id);

CREATE INDEX IF NOT EXISTS benchmarks_branch_idx ON benchmarks (branch);

CREATE TABLE IF NOT EXISTS device_measurements (
  measurement_id SERIAL PRIMARY KEY,
  benchmark_id int REFERENCES benchmarks (benchmark_id),
  cpu_util double precision,
  mem_megabytes double precision,
  gpu_util double precision,
  gpu_mem_megabytes double precision,
  time timestamp without time zone NOT NULL DEFAULT (current_timestamp AT TIME ZONE 'UTC')
);

CREATE INDEX IF NOT EXISTS device_measurements_branch_idx ON device_measurements (benchmark_id);

CREATE TABLE IF NOT EXISTS model_measurements (
  measurement_id SERIAL PRIMARY KEY,
  benchmark_id int REFERENCES benchmarks (benchmark_id),
  measurements jsonb,
  time timestamp without time zone NOT NULL DEFAULT (current_timestamp AT TIME ZONE 'UTC')
);

CREATE INDEX IF NOT EXISTS model_measurements_branch_idx ON model_measurements (benchmark_id);
