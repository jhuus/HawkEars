ALTER TABLE project ADD COLUMN recording_directory TEXT;
ALTER TABLE project ADD COLUMN recording_path_type TEXT
    CHECK (recording_path_type IS NULL OR recording_path_type IN ('absolute', 'project_relative'));
ALTER TABLE project ADD COLUMN recurse INTEGER NOT NULL DEFAULT 0
    CHECK (recurse IN (0, 1));
