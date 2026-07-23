ALTER TABLE review_queue ADD COLUMN location_max_score INTEGER
    NOT NULL DEFAULT 0 CHECK (location_max_score IN (0, 1));
