CREATE TABLE analysis_run_import (
    analysis_run_id INTEGER PRIMARY KEY
        REFERENCES analysis_run(id) ON DELETE CASCADE,
    import_batch_id INTEGER NOT NULL UNIQUE
        REFERENCES import_batch(id) ON DELETE CASCADE
);

CREATE TABLE imported_analysis_detection (
    detection_id INTEGER PRIMARY KEY
        REFERENCES detection(id) ON DELETE CASCADE,
    import_batch_id INTEGER NOT NULL
        REFERENCES import_batch(id) ON DELETE CASCADE,
    source_file TEXT NOT NULL,
    source_row INTEGER NOT NULL CHECK (source_row > 0),
    raw_recording TEXT,
    raw_species TEXT,
    raw_start TEXT,
    raw_end TEXT,
    raw_score TEXT
);

CREATE INDEX imported_analysis_detection_batch_idx
    ON imported_analysis_detection(import_batch_id);
