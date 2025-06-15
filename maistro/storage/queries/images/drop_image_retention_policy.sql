-- Remove any existing retention policy for images table
DO $$
BEGIN
    -- Try to remove the retention policy, ignore if it doesn't exist
    BEGIN
        PERFORM
            remove_retention_policy('images');
    EXCEPTION
        WHEN undefined_object THEN
            -- Policy doesn't exist, which is fine
    END;
END
$$;

