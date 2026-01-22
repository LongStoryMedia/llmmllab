// Type-safe parameter utilities for dynamic parameter optimization configuration
// This file provides utilities to work with performance parameters in a type-safe way
// that automatically adapts when new parameters are added to the schema.

import { PerformanceParameter } from '../types/PerformanceParameter';
import { ParameterTuningStrategy, ParameterTuningStrategyValues } from '../types/ParameterTuningStrategy';

// Type-safe parameter configuration with compile-time validation
// When new parameters are added to the YAML schema and types are regenerated,
// TypeScript will enforce that they are included here.
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

export const PARAMETER_CONFIGS: ParameterConfig = {
  n_ctx: {
    label: 'Context Size',
    description: 'Memory window for model attention',
    defaultPriority: 1,
    defaultStrategy: ParameterTuningStrategyValues.BINARY_SEARCH as ParameterTuningStrategy,
    defaultMaxAttempts: 10,
    defaultFloor: 8192,
    defaultOperator: '*',
    defaultModifier: 2,
    defaultMaxValue: 131072
  },
  n_batch: {
    label: 'Batch Size',
    description: 'Number of tokens processed together',
    defaultPriority: 3,
    defaultStrategy: ParameterTuningStrategyValues.CONSERVATIVE_INCREMENT as ParameterTuningStrategy,
    defaultMaxAttempts: 10,
    defaultFloor: 32,
    defaultOperator: '*',
    defaultModifier: 2,
    defaultMaxValue: 8192
  },
  n_ubatch: {
    label: 'Micro-batch',
    description: 'Internal batching for efficiency',
    defaultPriority: 4,
    defaultStrategy: ParameterTuningStrategyValues.CONSERVATIVE_INCREMENT as ParameterTuningStrategy,
    defaultMaxAttempts: 10,
    defaultFloor: 8,
    defaultOperator: '*',
    defaultModifier: 2,
    defaultMaxValue: 8192
  },
  n_gpu_layers: {
    label: 'GPU Layers',
    description: 'Layers to offload to GPU vs CPU',
    defaultPriority: 2,
    defaultStrategy: ParameterTuningStrategyValues.BINARY_SEARCH as ParameterTuningStrategy,
    defaultMaxAttempts: 10,
    defaultFloor: 1,
    defaultOperator: '+',
    defaultModifier: 10,
    defaultMaxValue: 125
  }
  // When you add new parameters to the YAML schema and regenerate types,
  // TypeScript will show a compile error here until you add the new parameter configuration.
  // This ensures no parameter is missed when extending the schema!
};

// Get all available parameter names (automatically updated when schema changes)
export const getAvailableParameterNames = (): PerformanceParameter['parameter_name'][] => {
  return Object.keys(PARAMETER_CONFIGS) as PerformanceParameter['parameter_name'][];
};

// Create parameter info for UI display
export const getParameterDisplayInfo = (paramName: PerformanceParameter['parameter_name']) => ({
  value: paramName,
  label: PARAMETER_CONFIGS[paramName].label,
  description: PARAMETER_CONFIGS[paramName].description
});

// Get all parameter display info for UI
export const getAllParameterDisplayInfo = () => {
  return getAvailableParameterNames().map(getParameterDisplayInfo);
};

// Create default parameter with type-safe configuration
export const createDefaultPerformanceParameter = (
  parameterName: PerformanceParameter['parameter_name']
): PerformanceParameter => {
  const config = PARAMETER_CONFIGS[parameterName];
  return {
    parameter_name: parameterName,
    priority: config.defaultPriority,
    tuning_strategy: config.defaultStrategy,
    max_search_attempts: config.defaultMaxAttempts,
    floor: config.defaultFloor,
    operator: config.defaultOperator,
    modifier: config.defaultModifier,
    max_value: config.defaultMaxValue
  };
};

// Validate that a parameter name is valid (runtime validation)
export const isValidParameterName = (name: string): name is PerformanceParameter['parameter_name'] => {
  return name in PARAMETER_CONFIGS;
};

// Get configuration for a parameter with runtime validation
export const getParameterConfig = (paramName: string) => {
  if (!isValidParameterName(paramName)) {
    throw new Error(`Invalid parameter name: ${paramName}. Valid parameters: ${getAvailableParameterNames().join(', ')}`);
  }
  return PARAMETER_CONFIGS[paramName];
};