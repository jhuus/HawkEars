CREATE TABLE review_queue (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL CHECK (length(trim(name)) > 0),
    analysis_run_id INTEGER NOT NULL
        REFERENCES analysis_run(id) ON DELETE CASCADE,
    species_id INTEGER NOT NULL REFERENCES species(id) ON DELETE RESTRICT,
    min_score REAL NOT NULL CHECK (min_score BETWEEN 0 AND 1),
    max_per_recording INTEGER NOT NULL CHECK (max_per_recording > 0),
    min_spacing_ms INTEGER NOT NULL CHECK (min_spacing_ms >= 0),
    ordering TEXT NOT NULL CHECK (ordering IN ('score', 'chronological')),
    created_at TEXT NOT NULL DEFAULT
        (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE review_queue_item (
    review_queue_id INTEGER NOT NULL
        REFERENCES review_queue(id) ON DELETE CASCADE,
    detection_id INTEGER NOT NULL REFERENCES detection(id) ON DELETE CASCADE,
    position INTEGER NOT NULL CHECK (position >= 0),
    PRIMARY KEY (review_queue_id, detection_id),
    UNIQUE (review_queue_id, position)
);

CREATE INDEX review_queue_run_idx ON review_queue(analysis_run_id);
CREATE INDEX review_queue_item_detection_idx ON review_queue_item(detection_id);
