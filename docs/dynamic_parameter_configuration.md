# Dynamic Parameter Configuration System

## Overview

The parameter optimization configuration system has been redesigned to be completely dynamic and self-maintaining. When new parameters are added to the YAML schema, the system automatically adapts without requiring manual updates to multiple files.

## Key Benefits

- **Automatic Schema Evolution**: When new parameters are added to `schemas/performance_parameter.yaml`, TypeScript will automatically enforce that they are configured properly
- **Compile-Time Safety**: Missing parameter configurations will cause TypeScript compilation errors, ensuring nothing is overlooked
- **Single Source of Truth**: All parameter configuration logic is centralized in `ui/src/utils/parameterUtils.ts`
- **Zero Manual Updates**: Frontend components automatically detect and display new parameters without code changes

## Architecture

### Core Files

The system consists of two main files:

#### ui/src/utils/parameterUtils.ts

This utility provides type-safe parameter management:

- Type-Safe Configuration using TypeScript mapped types to enforce complete parameter coverage
- Default Value Management with sensible defaults for all parameter fields
- Runtime Validation that validates parameter names and provides helpful error messages
- Display Utilities that generate labels and descriptions for UI components

#### Key Types

The system uses TypeScript mapped types to enforce completeness:

```typescript
// Enforces that ALL parameters from the schema must be configured
type ParameterConfig = {
  [K in PerformanceParameter['parameter_name']]: {
    label: string;
    description: string;
    defaultPriority: number;
    // ... other configuration fields
  };
};
```

### Automatic Detection Mechanism

When you add a new parameter to `schemas/performance_parameter.yaml`, the system automatically detects it:

- Schema Regeneration: Run `./regenerate_models.sh` to update TypeScript types
- Compilation Error: TypeScript will show errors in `PARAMETER_CONFIGS` for missing parameters
- Add Configuration: Simply add the new parameter configuration to fix the error
- Automatic UI: Frontend components will automatically display the new parameter

### Frontend Integration

#### ui/src/components/Settings/ParameterOptimizationSettings.tsx

The frontend component uses dynamic parameter loading:

- Uses `getAllParameterDisplayInfo()` to get current parameters
- Uses `createDefaultPerformanceParameter()` for new instances
- All parameter lists are generated dynamically with no hardcoded values

## Example: Adding a New Parameter

Here's how to add a new parameter to the system:

Update YAML Schema in `schemas/performance_parameter.yaml`:

```yaml
parameter_name:
  enum: ['n_ctx', 'n_batch', 'n_ubatch', 'n_gpu_layers', 'batch_size', 'new_parameter']
```

Regenerate Types:

```bash
./regenerate_models.sh
```

TypeScript Error will appear in `parameterUtils.ts`:

```text
Property 'new_parameter' is missing in type 'ParameterConfig'
```

Add Configuration in `ui/src/utils/parameterUtils.ts`:

```typescript
export const PARAMETER_CONFIGS: ParameterConfig = {
  // ... existing parameters
  new_parameter: {
    label: 'New Parameter',
    description: 'Description of the new parameter',
    defaultPriority: 6,
    defaultStrategy: ParameterTuningStrategyValues.CONSERVATIVE_INCREMENT,
    // ... other defaults
  }
};
```

The parameter automatically appears in all frontend components.

## Validation Features

The system includes comprehensive validation:

- **Runtime Type Checking**: `isValidParameterName()` validates parameter names at runtime
- **Error Messages**: Helpful error messages list all valid parameters when validation fails  
- **Compile-Time Enforcement**: TypeScript prevents missing or incorrect configurations

## Migration Benefits

The new system significantly reduces maintenance overhead:

**Before**: Adding a parameter required updating:

- Parameter configuration object
- Default values object  
- UI display constants
- Multiple frontend components

**After**: Adding a parameter requires only:

- YAML schema update
- Single configuration entry in `parameterUtils.ts`
- TypeScript enforces completeness automatically

## Future Extensibility

The system is designed to handle future enhancements:

- New parameter types can be added to the YAML schema
- Configuration properties can be extended in the utility types
- UI components automatically adapt to new parameters
- No breaking changes required for additions

This architecture ensures that the parameter optimization system remains maintainable and extensible as the application grows.