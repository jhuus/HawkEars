-- Imported reviewed labels may have no model confidence for their species.
-- The migration runner disables foreign keys during this atomic table rebuild
-- and checks all relationships before committing.
CREATE TABLE detection_new (
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
        (source = 'inference' AND analysis_item_id IS NOT NULL AND import_batch_id IS NULL)
        OR (source = 'manual' AND analysis_item_id IS NULL AND import_batch_id IS NULL)
        OR (source = 'import' AND analysis_item_id IS NULL AND import_batch_id IS NOT NULL)
    )
);
INSERT INTO detection_new SELECT * FROM detection;
DROP TABLE detection;
ALTER TABLE detection_new RENAME TO detection;

CREATE INDEX detection_recording_idx ON detection(recording_id);
CREATE INDEX detection_analysis_item_idx ON detection(analysis_item_id);
CREATE INDEX detection_import_batch_idx ON detection(import_batch_id);
CREATE INDEX detection_source_score_idx ON detection(source, score);
CREATE INDEX detection_current_revision_idx ON detection(current_revision_id);

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
