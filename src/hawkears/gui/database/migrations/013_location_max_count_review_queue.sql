ALTER TABLE review_queue ADD COLUMN location_max_count INTEGER NOT NULL DEFAULT 0
    CHECK (location_max_count IN (0, 1));
