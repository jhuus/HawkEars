ALTER TABLE review_queue ADD COLUMN confirmation_scope TEXT NOT NULL DEFAULT 'none'
    CHECK (confirmation_scope IN ('none', 'recording', 'location', 'location_date'));

ALTER TABLE review_queue_item ADD COLUMN confirmation_key TEXT;

ALTER TABLE review_queue_item ADD COLUMN state TEXT NOT NULL DEFAULT 'pending'
    CHECK (state IN ('pending', 'reviewed', 'skipped'));

ALTER TABLE review_queue_item ADD COLUMN skipped_by_detection_id INTEGER
    REFERENCES detection(id) ON DELETE SET NULL;

CREATE INDEX review_queue_item_state_idx
    ON review_queue_item(review_queue_id, state);

UPDATE review_queue
SET confirmation_scope = 'location'
WHERE location_max_count = 1
   OR location_max_score_sum = 1
   OR location_max_score = 1
   OR location_first_date = 1;

UPDATE review_queue
SET confirmation_scope = 'location_date'
WHERE location_date_high_score = 1
   OR location_date_first_detection = 1;

UPDATE review_queue_item
SET confirmation_key = (
    SELECT CASE review_queue.confirmation_scope
        WHEN 'recording' THEN CAST(detection.recording_id AS TEXT)
        WHEN 'location' THEN
            CASE WHEN coalesce(
                nullif(analysis_item.location_name, ''),
                nullif(recording.location_name, ''),
                nullif(analysis_item.region_code, ''),
                nullif(recording.region_code, ''),
                nullif(json_extract(analysis_run.settings_json,
                    '$.location.region_code'), '')
            ) IS NULL THEN NULL
            ELSE coalesce(
                nullif(analysis_item.location_name, ''),
                nullif(recording.location_name, ''),
                nullif(analysis_item.region_code, ''),
                nullif(recording.region_code, ''),
                nullif(json_extract(analysis_run.settings_json,
                    '$.location.region_code'), '')
            ) END
        WHEN 'location_date' THEN
            CASE WHEN coalesce(
                nullif(analysis_item.location_name, ''),
                nullif(recording.location_name, ''),
                nullif(analysis_item.region_code, ''),
                nullif(recording.region_code, ''),
                nullif(json_extract(analysis_run.settings_json,
                    '$.location.region_code'), '')
            ) IS NULL OR coalesce(
                substr(analysis_item.recorded_at, 1, 10),
                substr(recording.recorded_at, 1, 10),
                CASE WHEN json_extract(analysis_run.settings_json,
                    '$.location.date_mode') = 'specific'
                THEN json_extract(analysis_run.settings_json,
                    '$.location.date') END
            ) IS NULL THEN NULL
            ELSE json_array(
                coalesce(
                    nullif(analysis_item.location_name, ''),
                    nullif(recording.location_name, ''),
                    nullif(analysis_item.region_code, ''),
                    nullif(recording.region_code, ''),
                    nullif(json_extract(analysis_run.settings_json,
                        '$.location.region_code'), '')
                ),
                coalesce(
                    substr(analysis_item.recorded_at, 1, 10),
                    substr(recording.recorded_at, 1, 10),
                    CASE WHEN json_extract(analysis_run.settings_json,
                        '$.location.date_mode') = 'specific'
                    THEN json_extract(analysis_run.settings_json,
                        '$.location.date') END
                )
            ) END
        ELSE NULL
    END
    FROM review_queue
    JOIN detection ON detection.id = review_queue_item.detection_id
    JOIN analysis_item ON analysis_item.id = detection.analysis_item_id
    JOIN analysis_run ON analysis_run.id = analysis_item.analysis_run_id
    JOIN recording ON recording.id = detection.recording_id
    WHERE review_queue.id = review_queue_item.review_queue_id
)
WHERE EXISTS (
    SELECT 1 FROM review_queue
    WHERE review_queue.id = review_queue_item.review_queue_id
      AND review_queue.confirmation_scope != 'none'
);

UPDATE review_queue_item
SET state = 'reviewed'
WHERE EXISTS (
    SELECT 1 FROM review
    WHERE review.detection_id = review_queue_item.detection_id
);
