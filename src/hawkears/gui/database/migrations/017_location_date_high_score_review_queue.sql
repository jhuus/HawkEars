ALTER TABLE review_queue ADD COLUMN location_date_high_score INTEGER
    NOT NULL DEFAULT 0 CHECK (location_date_high_score IN (0, 1));
