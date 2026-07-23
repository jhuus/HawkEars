ALTER TABLE review_queue ADD COLUMN duration_ranked INTEGER NOT NULL DEFAULT 0
    CHECK (duration_ranked IN (0, 1));
