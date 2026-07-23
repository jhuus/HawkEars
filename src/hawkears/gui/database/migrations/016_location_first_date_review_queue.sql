ALTER TABLE review_queue ADD COLUMN location_first_date INTEGER
    NOT NULL DEFAULT 0 CHECK (location_first_date IN (0, 1));
