/**
 * User preferences
 */
export interface PreferencesConfig {
    /**
     * Default profile ID
     */
    default_profile_id?: string;
    /**
     * Theme
     */
    theme?: string;
    /**
     * Font size
     */
    font_size?: number;
    /**
     * Notifications enabled
     */
    notifications_on?: boolean;
    /**
     * Language
     */
    language?: string;
}
