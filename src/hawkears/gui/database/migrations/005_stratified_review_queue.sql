ALTER TABLE review_queue ADD COLUMN score_band_width REAL
    CHECK (score_band_width IS NULL OR score_band_width > 0);

ALTER TABLE review_queue ADD COLUMN max_per_score_band INTEGER
    CHECK (max_per_score_band IS NULL OR max_per_score_band > 0);
