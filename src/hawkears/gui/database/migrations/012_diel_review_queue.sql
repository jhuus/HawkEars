ALTER TABLE review_queue ADD COLUMN diel_bin_count INTEGER
    CHECK (diel_bin_count IS NULL OR diel_bin_count BETWEEN 2 AND 24);

ALTER TABLE review_queue ADD COLUMN max_per_diel_bin INTEGER
    CHECK (max_per_diel_bin IS NULL OR max_per_diel_bin > 0);
