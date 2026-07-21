CREATE TABLE project (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    name TEXT NOT NULL CHECK (length(trim(name)) > 0),
    description TEXT NOT NULL DEFAULT '',
    analysis_settings_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(analysis_settings_json)),
    format_version INTEGER NOT NULL DEFAULT 1 CHECK (format_version > 0),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE species (
    id INTEGER PRIMARY KEY,
    canonical_key TEXT UNIQUE,
    class_name TEXT,
    common_name TEXT NOT NULL CHECK (length(trim(common_name)) > 0),
    scientific_name TEXT,
    species_code TEXT,
    ebird_code TEXT,
    model_class_index INTEGER CHECK (model_class_index IS NULL OR model_class_index >= 0),
    source TEXT NOT NULL DEFAULT 'custom'
        CHECK (source IN ('hawkears', 'ebird', 'custom')),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX species_common_name_idx ON species(common_name COLLATE NOCASE);
CREATE INDEX species_scientific_name_idx ON species(scientific_name COLLATE NOCASE);
CREATE UNIQUE INDEX species_ebird_code_idx
    ON species(ebird_code) WHERE ebird_code IS NOT NULL;

CREATE TABLE project_species (
    project_id INTEGER NOT NULL DEFAULT 1 REFERENCES project(id) ON DELETE CASCADE,
    species_id INTEGER NOT NULL REFERENCES species(id) ON DELETE RESTRICT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (project_id, species_id)
);

CREATE TABLE recording (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL CHECK (length(path) > 0),
    path_type TEXT NOT NULL CHECK (path_type IN ('absolute', 'project_relative')),
    display_name TEXT NOT NULL CHECK (length(trim(display_name)) > 0),
    duration_ms INTEGER CHECK (duration_ms IS NULL OR duration_ms >= 0),
    sample_rate INTEGER CHECK (sample_rate IS NULL OR sample_rate > 0),
    channels INTEGER CHECK (channels IS NULL OR channels > 0),
    recorded_at TEXT,
    latitude REAL CHECK (latitude IS NULL OR latitude BETWEEN -90 AND 90),
    longitude REAL CHECK (longitude IS NULL OR longitude BETWEEN -180 AND 180),
    region_code TEXT,
    location_name TEXT,
    file_size INTEGER CHECK (file_size IS NULL OR file_size >= 0),
    modified_ns INTEGER CHECK (modified_ns IS NULL OR modified_ns >= 0),
    fingerprint TEXT,
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (path, path_type)
);

CREATE INDEX recording_recorded_at_idx ON recording(recorded_at);
CREATE INDEX recording_region_code_idx ON recording(region_code);

CREATE TABLE analysis_run (
    id INTEGER PRIMARY KEY,
    name TEXT,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'cancelled', 'failed')),
    hawkears_version TEXT NOT NULL,
    model_version TEXT,
    settings_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(settings_json)),
    started_at TEXT,
    finished_at TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    error_message TEXT
);

CREATE TABLE analysis_run_species (
    analysis_run_id INTEGER NOT NULL REFERENCES analysis_run(id) ON DELETE CASCADE,
    species_id INTEGER NOT NULL REFERENCES species(id) ON DELETE RESTRICT,
    PRIMARY KEY (analysis_run_id, species_id)
);

CREATE TABLE analysis_item (
    id INTEGER PRIMARY KEY,
    analysis_run_id INTEGER NOT NULL REFERENCES analysis_run(id) ON DELETE CASCADE,
    recording_id INTEGER NOT NULL REFERENCES recording(id) ON DELETE RESTRICT,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'skipped', 'cancelled', 'failed')),
    started_at TEXT,
    finished_at TEXT,
    error_message TEXT,
    processing_seconds REAL CHECK (processing_seconds IS NULL OR processing_seconds >= 0),
    UNIQUE (analysis_run_id, recording_id)
);

CREATE INDEX analysis_item_recording_idx ON analysis_item(recording_id);

CREATE TABLE import_batch (
    id INTEGER PRIMARY KEY,
    provider TEXT NOT NULL CHECK (length(trim(provider)) > 0),
    source_path TEXT,
    format_version TEXT,
    model_name TEXT,
    model_version TEXT,
    settings_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(settings_json)),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'completed', 'partial', 'failed')),
    error_message TEXT,
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE detection (
    id INTEGER PRIMARY KEY,
    recording_id INTEGER NOT NULL REFERENCES recording(id) ON DELETE CASCADE,
    analysis_item_id INTEGER REFERENCES analysis_item(id) ON DELETE CASCADE,
    import_batch_id INTEGER REFERENCES import_batch(id) ON DELETE CASCADE,
    source TEXT NOT NULL CHECK (source IN ('inference', 'manual', 'import')),
    score REAL CHECK (score IS NULL OR score BETWEEN 0 AND 1),
    current_revision_id INTEGER REFERENCES detection_revision(id) ON DELETE SET NULL,
    created_by TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CHECK (
        (source = 'inference' AND analysis_item_id IS NOT NULL AND import_batch_id IS NULL AND score IS NOT NULL)
        OR (source = 'manual' AND analysis_item_id IS NULL AND import_batch_id IS NULL)
        OR (source = 'import' AND analysis_item_id IS NULL AND import_batch_id IS NOT NULL)
    )
);

CREATE INDEX detection_recording_idx ON detection(recording_id);
CREATE INDEX detection_analysis_item_idx ON detection(analysis_item_id);
CREATE INDEX detection_import_batch_idx ON detection(import_batch_id);
CREATE INDEX detection_source_score_idx ON detection(source, score);

CREATE TABLE detection_revision (
    id INTEGER PRIMARY KEY,
    detection_id INTEGER NOT NULL REFERENCES detection(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL CHECK (revision_number > 0),
    species_id INTEGER NOT NULL REFERENCES species(id) ON DELETE RESTRICT,
    start_ms INTEGER NOT NULL CHECK (start_ms >= 0),
    end_ms INTEGER NOT NULL CHECK (end_ms > start_ms),
    low_frequency_hz INTEGER,
    high_frequency_hz INTEGER,
    change_notes TEXT NOT NULL DEFAULT '',
    created_by TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (detection_id, revision_number),
    CHECK (
        (low_frequency_hz IS NULL AND high_frequency_hz IS NULL)
        OR (
            low_frequency_hz IS NOT NULL
            AND high_frequency_hz IS NOT NULL
            AND low_frequency_hz >= 0
            AND high_frequency_hz > low_frequency_hz
        )
    )
);

CREATE INDEX detection_revision_species_idx ON detection_revision(species_id);

CREATE TRIGGER detection_current_revision_guard
BEFORE UPDATE OF current_revision_id ON detection
WHEN NEW.current_revision_id IS NOT NULL
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM detection_revision
        WHERE id = NEW.current_revision_id AND detection_id = NEW.id
    ) THEN RAISE(ABORT, 'current revision belongs to another detection') END;
END;

CREATE TRIGGER inference_recording_guard
BEFORE INSERT ON detection
WHEN NEW.source = 'inference'
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM analysis_item
        WHERE id = NEW.analysis_item_id AND recording_id = NEW.recording_id
    ) THEN RAISE(ABORT, 'analysis item belongs to another recording') END;
END;

CREATE TABLE import_detection (
    detection_id INTEGER PRIMARY KEY REFERENCES detection(id) ON DELETE CASCADE,
    raw_recording TEXT,
    raw_species TEXT,
    raw_start TEXT,
    raw_end TEXT,
    raw_score TEXT,
    raw_data_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(raw_data_json)),
    source_row INTEGER CHECK (source_row IS NULL OR source_row > 0)
);

CREATE TABLE review (
    id INTEGER PRIMARY KEY,
    detection_id INTEGER NOT NULL UNIQUE REFERENCES detection(id) ON DELETE CASCADE,
    verdict TEXT NOT NULL CHECK (verdict IN ('correct', 'incorrect', 'uncertain')),
    notes TEXT NOT NULL DEFAULT '',
    reviewer TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX review_verdict_idx ON review(verdict);

CREATE TABLE detection_additional_species (
    detection_id INTEGER NOT NULL REFERENCES detection(id) ON DELETE CASCADE,
    species_id INTEGER NOT NULL REFERENCES species(id) ON DELETE RESTRICT,
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (detection_id, species_id)
);
