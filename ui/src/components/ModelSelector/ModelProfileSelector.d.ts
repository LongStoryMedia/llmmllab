import { ModelProfileConfig } from "../../types/ModelProfileConfig";
declare const ModelProfileSelector: ({ task }: {
    task: {
        key: keyof ModelProfileConfig;
        label: string;
    };
}) => import("react/jsx-runtime").JSX.Element;
export default ModelProfileSelector;
