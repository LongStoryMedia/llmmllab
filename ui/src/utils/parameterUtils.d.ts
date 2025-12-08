import { PerformanceParameter } from '../types/PerformanceParameter';
import { ParameterTuningStrategy } from '../types/ParameterTuningStrategy';
type ParameterConfig = {
    [K in PerformanceParameter['parameter_name']]: {
        label: string;
        description: string;
        defaultPriority: number;
        defaultStrategy: ParameterTuningStrategy;
        defaultMaxAttempts: number;
        defaultFloor: number;
        defaultOperator: PerformanceParameter['operator'];
        defaultModifier: number;
        defaultMaxValue: number;
    };
};
export declare const PARAMETER_CONFIGS: ParameterConfig;
export declare const getAvailableParameterNames: () => PerformanceParameter["parameter_name"][];
export declare const getParameterDisplayInfo: (paramName: PerformanceParameter["parameter_name"]) => {
    value: "n_ctx" | "n_batch" | "n_ubatch" | "n_gpu_layers";
    label: string;
    description: string;
};
export declare const getAllParameterDisplayInfo: () => {
    value: "n_ctx" | "n_batch" | "n_ubatch" | "n_gpu_layers";
    label: string;
    description: string;
}[];
export declare const createDefaultPerformanceParameter: (parameterName: PerformanceParameter["parameter_name"]) => PerformanceParameter;
export declare const isValidParameterName: (name: string) => name is PerformanceParameter["parameter_name"];
export declare const getParameterConfig: (paramName: string) => {
    label: string;
    description: string;
    defaultPriority: number;
    defaultStrategy: ParameterTuningStrategy;
    defaultMaxAttempts: number;
    defaultFloor: number;
    defaultOperator: PerformanceParameter["operator"];
    defaultModifier: number;
    defaultMaxValue: number;
};
export {};
