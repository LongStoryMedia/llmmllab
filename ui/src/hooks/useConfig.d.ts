import { UserConfig } from '../types/UserConfig';
export declare function useConfig(): {
    config: UserConfig | null;
    isLoading: boolean;
    error: Error | null;
    fetchConfig: () => Promise<void>;
    updateConfig: import("react").Dispatch<import("react").SetStateAction<UserConfig | null>>;
    updatePartialConfig: (section: keyof UserConfig, sectionConfig: unknown) => Promise<boolean>;
};
