import { useState, useEffect } from 'react';
import {
  Grid,
  Typography,
  TextField,
  Switch,
  FormControlLabel,
  Button,
  Paper,
  Alert,
  Box
} from '@mui/material';
import { CircuitBreakerConfig } from '../../types/CircuitBreakerConfig';
import { useConfig } from '../../hooks/useConfig';

const CircuitBreakerSettings = () => {
  const { config, updatePartialConfig, isLoading, error } = useConfig();
  const [settings, setSettings] = useState<CircuitBreakerConfig>({
    base_timeout: 60.0,
    deep_research_timeout: 120.0,
    max_retries: 2,
    cooldown_period: 30.0,
    enable_perplexity_guard: true,
    perplexity_window: 40,
    perplexity_threshold: 10.0,
    avg_logprob_floor: -6.0,
    repetition_ngram: 6,
    repetition_threshold: 6,
    min_tokens_for_eval: 20,
    perplexity_log_interval_tokens: 20,
    log_repetition_events: true,
    tool_gen_repetition_ngram: 4,
    tool_gen_repetition_threshold: 3
  });
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    if (config?.circuit_breaker) {
      setSettings(config.circuit_breaker);
      setHasChanges(false);
    }
  }, [config]);

  const handleChange = (field: keyof CircuitBreakerConfig, value: string | number | boolean) => {
    setSettings(prev => ({ ...prev, [field]: value }));
    setHasChanges(true);
  };

  const handleSave = async () => {
    if (!config) {
      return;
    }

    try {
      const success = await updatePartialConfig('circuit_breaker', settings);
      if (success) {
        setHasChanges(false);
      } else {
        console.error('Failed to save circuit breaker settings');
      }
    } catch (err) {
      console.error('Failed to save circuit breaker settings:', err);
    }
  };

  const handleReset = () => {
    if (config?.circuit_breaker) {
      setSettings(config.circuit_breaker);
      setHasChanges(false);
    }
  };

  if (isLoading) {
    return <Typography>Loading circuit breaker settings...</Typography>;
  }

  if (error) {
    return <Alert severity="error">Error loading circuit breaker settings: {error.message}</Alert>;
  }

  return (
    <Grid container spacing={3}>
      <Grid size={12}>
        <Typography variant="h6" gutterBottom>
          Circuit Breaker & Quality Control Settings
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Configure timeout protection, retry behavior, and quality monitoring for AI models.
        </Typography>
      </Grid>

      {/* Timeout Settings */}
      <Grid size={12}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Timeout Settings
          </Typography>

          <Grid container spacing={2}>
            <Grid size={6}>
              <TextField
                fullWidth
                label="Base Timeout (seconds)"
                type="number"
                value={settings.base_timeout}
                onChange={(e) => handleChange('base_timeout', parseFloat(e.target.value))}
                inputProps={{ min: 1, max: 600, step: 1 }}
                helperText="Standard timeout for model operations"
              />
            </Grid>
            <Grid size={6}>
              <TextField
                fullWidth
                label="Deep Research Timeout (seconds)"
                type="number"
                value={settings.deep_research_timeout}
                onChange={(e) => handleChange('deep_research_timeout', parseFloat(e.target.value))}
                inputProps={{ min: 1, max: 1200, step: 1 }}
                helperText="Extended timeout for complex research tasks"
              />
            </Grid>
          </Grid>
        </Paper>
      </Grid>

      {/* Retry Settings */}
      <Grid size={12}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Retry Settings
          </Typography>

          <Grid container spacing={2}>
            <Grid size={6}>
              <TextField
                fullWidth
                label="Max Retries"
                type="number"
                value={settings.max_retries}
                onChange={(e) => handleChange('max_retries', parseInt(e.target.value))}
                inputProps={{ min: 0, max: 10, step: 1 }}
                helperText="Maximum number of retry attempts"
              />
            </Grid>
            <Grid size={6}>
              <TextField
                fullWidth
                label="Cooldown Period (seconds)"
                type="number"
                value={settings.cooldown_period}
                onChange={(e) => handleChange('cooldown_period', parseFloat(e.target.value))}
                inputProps={{ min: 0, max: 300, step: 1 }}
                helperText="Wait time before allowing retry after failure"
              />
            </Grid>
          </Grid>
        </Paper>
      </Grid>

      {/* Perplexity Guard Settings */}
      <Grid size={12}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Perplexity Guard Settings
          </Typography>

          <FormControlLabel
            control={
              <Switch
                checked={settings.enable_perplexity_guard}
                onChange={(e) => handleChange('enable_perplexity_guard', e.target.checked)}
              />
            }
            label="Enable Perplexity-Based Quality Monitoring"
            sx={{ mb: 2 }}
          />

          {settings.enable_perplexity_guard && (
            <Grid container spacing={2}>
              <Grid size={6}>
                <TextField
                  fullWidth
                  label="Perplexity Window"
                  type="number"
                  value={settings.perplexity_window}
                  onChange={(e) => handleChange('perplexity_window', parseInt(e.target.value))}
                  inputProps={{ min: 10, max: 200, step: 1 }}
                  helperText="Number of tokens to consider for perplexity calculation"
                />
              </Grid>
              <Grid size={6}>
                <TextField
                  fullWidth
                  label="Perplexity Threshold"
                  type="number"
                  value={settings.perplexity_threshold}
                  onChange={(e) => handleChange('perplexity_threshold', parseFloat(e.target.value))}
                  inputProps={{ min: 1.0, max: 50.0, step: 0.1 }}
                  helperText="Threshold above which to trigger quality concerns"
                />
              </Grid>
              <Grid size={6}>
                <TextField
                  fullWidth
                  label="Average Log Probability Floor"
                  type="number"
                  value={settings.avg_logprob_floor}
                  onChange={(e) => handleChange('avg_logprob_floor', parseFloat(e.target.value))}
                  inputProps={{ min: -20.0, max: 0.0, step: 0.1 }}
                  helperText="Minimum average log probability threshold"
                />
              </Grid>
              <Grid size={6}>
                <TextField
                  fullWidth
                  label="Min Tokens for Evaluation"
                  type="number"
                  value={settings.min_tokens_for_eval}
                  onChange={(e) => handleChange('min_tokens_for_eval', parseInt(e.target.value))}
                  inputProps={{ min: 5, max: 500, step: 1 }}
                  helperText="Minimum tokens before starting quality evaluation"
                />
              </Grid>
            </Grid>
          )}
        </Paper>
      </Grid>

      {/* Repetition Detection Settings */}
      <Grid size={12}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Repetition Detection Settings
          </Typography>

          <Grid container spacing={2}>
            <Grid size={6}>
              <TextField
                fullWidth
                label="Repetition N-gram Size"
                type="number"
                value={settings.repetition_ngram}
                onChange={(e) => handleChange('repetition_ngram', parseInt(e.target.value))}
                inputProps={{ min: 2, max: 20, step: 1 }}
                helperText="N-gram size for repetition detection"
              />
            </Grid>
            <Grid size={6}>
              <TextField
                fullWidth
                label="Repetition Threshold"
                type="number"
                value={settings.repetition_threshold}
                onChange={(e) => handleChange('repetition_threshold', parseInt(e.target.value))}
                inputProps={{ min: 2, max: 20, step: 1 }}
                helperText="Number of repetitions before triggering detection"
              />
            </Grid>
            <Grid size={6}>
              <TextField
                fullWidth
                label="Tool Generation N-gram Size"
                type="number"
                value={settings.tool_gen_repetition_ngram}
                onChange={(e) => handleChange('tool_gen_repetition_ngram', parseInt(e.target.value))}
                inputProps={{ min: 2, max: 20, step: 1 }}
                helperText="N-gram size for tool generation repetition detection"
              />
            </Grid>
            <Grid size={6}>
              <TextField
                fullWidth
                label="Tool Generation Threshold"
                type="number"
                value={settings.tool_gen_repetition_threshold}
                onChange={(e) => handleChange('tool_gen_repetition_threshold', parseInt(e.target.value))}
                inputProps={{ min: 2, max: 20, step: 1 }}
                helperText="Repetition threshold for tool generation"
              />
            </Grid>
          </Grid>
        </Paper>
      </Grid>

      {/* Logging Settings */}
      <Grid size={12}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Logging Settings
          </Typography>

          <Grid container spacing={2}>
            <Grid size={6}>
              <TextField
                fullWidth
                label="Perplexity Log Interval (tokens)"
                type="number"
                value={settings.perplexity_log_interval_tokens}
                onChange={(e) => handleChange('perplexity_log_interval_tokens', parseInt(e.target.value))}
                inputProps={{ min: 5, max: 100, step: 1 }}
                helperText="Interval in tokens for logging perplexity metrics"
              />
            </Grid>
            <Grid size={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.log_repetition_events}
                    onChange={(e) => handleChange('log_repetition_events', e.target.checked)}
                  />
                }
                label="Log Repetition Events"
                sx={{ mt: 2 }}
              />
            </Grid>
          </Grid>
        </Paper>
      </Grid>

      {/* Action Buttons */}
      <Grid size={12}>
        <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
          <Button
            variant="outlined"
            onClick={handleReset}
            disabled={!hasChanges}
          >
            Reset
          </Button>
          <Button
            variant="contained"
            onClick={handleSave}
            disabled={!hasChanges || isLoading}
          >
            Save Changes
          </Button>
        </Box>
      </Grid>
    </Grid>
  );
};

export default CircuitBreakerSettings;