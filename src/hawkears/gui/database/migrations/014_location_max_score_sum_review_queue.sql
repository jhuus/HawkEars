ALTER TABLE review_queue ADD COLUMN location_max_score_sum INTEGER
    NOT NULL DEFAULT 0 CHECK (location_max_score_sum IN (0, 1));
