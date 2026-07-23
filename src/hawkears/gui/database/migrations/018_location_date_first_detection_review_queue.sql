ALTER TABLE review_queue ADD COLUMN location_date_first_detection INTEGER
    NOT NULL DEFAULT 0 CHECK (location_date_first_detection IN (0, 1));
