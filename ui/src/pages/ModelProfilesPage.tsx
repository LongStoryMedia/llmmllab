import { useEffect, useState } from 'react';
import {
  Box, Typography, Button, Dialog, DialogTitle, DialogContent, DialogActions,
  TextField, Paper, IconButton, Grid, FormControl, InputLabel, Select, MenuItem,
  Chip, FormControlLabel, Checkbox, Switch, Slider, Accordion, AccordionSummary,
  AccordionDetails, Alert
} from '@mui/material';
import {
  Delete as DeleteIcon, Edit as EditIcon, Add as AddIcon,
  ExpandMore as ExpandMoreIcon, Memory as MemoryIcon,
  Warning as WarningIcon, Settings as SettingsIcon
} from '@mui/icons-material';
import { listModelProfiles, createModelProfile, updateModelProfile, deleteModelProfile } from '../api/model';
import { ModelProfile } from '../types/ModelProfile';
import { GPUConfig } from '../types/GpuConfig';
import { useAuth } from '../auth';
import ModelSelector from '../components/ModelSelector/ModelSelector';
import { getToken } from '../api';
import { ModelProfileType } from '../types/ModelProfileType';

const getModelProfileTypeName = (type: ModelProfileType): string => {
  switch (type) {
    case ModelProfileType.Primary: return 'Primary';
    case ModelProfileType.PrimarySummary: return 'Primary Summary';
    case ModelProfileType.MasterSummary: return 'Master Summary';
    case ModelProfileType.BriefSummary: return 'Brief Summary';
    case ModelProfileType.KeyPoints: return 'Key Points';
    case ModelProfileType.SelfCritique: return 'Self Critique';
    case ModelProfileType.Improvement: return 'Improvement';
    case ModelProfileType.MemoryRetrieval: return 'Memory Retrieval';
    case ModelProfileType.Analysis: return 'Analysis';
    case ModelProfileType.ResearchTask: return 'Research Task';
    case ModelProfileType.ResearchPlan: return 'Research Plan';
    case ModelProfileType.ResearchConsolidation: return 'Research Consolidation';
    case ModelProfileType.ResearchAnalysis: return 'Research Analysis';
    case ModelProfileType.Embedding: return 'Embedding';
    case ModelProfileType.Formatting: return 'Formatting';
    case ModelProfileType.ImageGenerationPrompt: return 'Image Generation Prompt';
    case ModelProfileType.Engineering: return 'Engineering';
    case ModelProfileType.Reranking: return 'Reranking';
    case ModelProfileType.ImageGeneration: return 'Image Generation';
    default: return 'Unknown';
  }
};

const emptyProfile: ModelProfile = {
  id: '',
  user_id: '',
  name: '',
  description: '',
  model_name: '',
  parameters: {
    think: false
  },
  system_prompt: '',
  created_at: new Date(),
  updated_at: new Date(),
  type: ModelProfileType.Primary
};

const ModelProfilesPage = () => {
  const [profiles, setProfiles] = useState<ModelProfile[]>([]);
  const [editingProfile, setEditingProfile] = useState<ModelProfile>(emptyProfile);
  const [dialogOpen, setDialogOpen] = useState(false);
  const auth = useAuth();

  // Fetch profiles on mount
  useEffect(() => {
    const fetchProfiles = async () => {
      try {
        // You may need to pass the token here
        const data = await listModelProfiles(getToken(auth.user));
        setProfiles(data);
      } catch (err: unknown) {
        if (err instanceof Error) {
          console.error('Error fetching model profiles:', err.message);
        }
      }
    };
    fetchProfiles();
  }, [auth.user]);

  // Handle add/edit profile
  const handleSaveProfile = async (isNew: boolean = false) => {
    const token = getToken(auth.user);
    if (editingProfile?.id && !isNew) {
      await updateModelProfile(token, editingProfile.id, editingProfile);
    } else {
      if (!editingProfile) {
        return;
      }
      await createModelProfile(token, editingProfile);
    }
    setDialogOpen(false);
    setEditingProfile(emptyProfile);
    // Refresh list
    const data = await listModelProfiles(token);
    setProfiles(data);
  };

  // Handle delete
  const handleDeleteProfile = async (id: string) => {
    const token = getToken(auth.user);
    await deleteModelProfile(token, id);
    setProfiles(profiles.filter(p => p.id !== id));
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>Model Profiles</Typography>
      <Button variant="contained" onClick={() => {
        setEditingProfile(emptyProfile); setDialogOpen(true);
      }}>Add Profile</Button>
      <Grid container spacing={2} sx={{ mt: 2, display: 'flex', flexDirection: 'column' }}>
        {profiles && profiles.map(profile => (
          <Grid key={profile.id} sx={{ p: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
            <Paper sx={{ p: 2, textAlign: 'left', width: '100%', display: 'flex', justifyContent: 'space-between' }}>
              <Box>
                <Typography variant="subtitle1">{profile.name}</Typography>
                <Typography variant="body2">{profile.description}</Typography>
                <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                  <Chip
                    label={getModelProfileTypeName(profile.type)}
                    size="small"
                    variant="outlined"
                    sx={{ mt: 1 }}
                  />
                  {profile.parameters?.think && (
                    <Chip
                      label="Think Mode"
                      size="small"
                      color="primary"
                      variant="outlined"
                      sx={{ mt: 1 }}
                    />
                  )}
                </Box>
              </Box>
              <Box>
                <IconButton onClick={() => {
                  // Ensure think field is properly set when editing
                  setEditingProfile({
                    ...profile,
                    parameters: {
                      ...profile.parameters,
                      think: profile.parameters?.think ?? false
                    }
                  });
                  setDialogOpen(true);
                }}><EditIcon /></IconButton>
                <IconButton onClick={() => profile.id && handleDeleteProfile(profile.id)}><DeleteIcon /></IconButton>
              </Box>
            </Paper>
          </Grid>
        ))}
      </Grid>
      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>{editingProfile?.id ? 'Edit Profile' : 'Add Profile'}</DialogTitle>
        <DialogContent>
          <TextField
            label="Name"
            value={editingProfile?.name || ''}
            onChange={e => setEditingProfile({ ...editingProfile, name: e.target.value })}
            fullWidth margin="normal"
          />
          <TextField
            label="Description"
            value={editingProfile?.description || ''}
            onChange={e => setEditingProfile({ ...editingProfile, description: e.target.value })}
            fullWidth margin="normal"
          />
          <FormControl fullWidth margin="normal">
            <InputLabel>Profile Type</InputLabel>
            <Select
              value={editingProfile?.type ?? ModelProfileType.Primary}
              onChange={e => setEditingProfile({ ...editingProfile, type: e.target.value as ModelProfileType })}
              label="Profile Type"
            >
              {Object.values(ModelProfileType).filter(v => typeof v === 'number').map((type) => (
                <MenuItem key={type} value={type as ModelProfileType}>
                  {getModelProfileTypeName(type as ModelProfileType)}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <ModelSelector
            onSelect={e => setEditingProfile({ ...editingProfile, model_name: e.target.value })}
            name={editingProfile?.model_name || ''}
          />
          <TextField
            label="System Prompt"
            value={editingProfile?.system_prompt || ''}
            onChange={e => setEditingProfile({ ...editingProfile, system_prompt: e.target.value })}
            fullWidth margin="normal"
            multiline
            minRows={2}
          />

          {/* Think Mode */}
          <FormControlLabel
            control={
              <Switch
                checked={editingProfile?.parameters?.think ?? false}
                onChange={(e) => setEditingProfile({
                  ...editingProfile,
                  parameters: {
                    ...editingProfile?.parameters,
                    think: e.target.checked
                  }
                })}
              />
            }
            label="Enable Think Mode"
            sx={{ mb: 1, display: 'block' }}
          />
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2, mt: -1 }}>
            When enabled, the model will show its internal reasoning process and thoughts before providing the final answer.
          </Typography>

          <TextField
            label="Number of Context"
            value={editingProfile?.parameters?.num_ctx || ''}
            onChange={e => setEditingProfile({ ...editingProfile, parameters: { ...editingProfile.parameters, num_ctx: Number(e.target.value) } })}
            fullWidth margin="normal"
            type="number"
            helperText="Sets the size of the context window used to generate the next token. (Default: 2048)"
          />
          <TextField
            label="Repeat Last N"
            value={editingProfile?.parameters?.repeat_last_n || ''}
            onChange={e => setEditingProfile({ ...editingProfile, parameters: { ...editingProfile.parameters, repeat_last_n: Number(e.target.value) } })}
            fullWidth margin="normal"
            type="number"
            helperText="	Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)"
          />
          <TextField
            label="Repeat Penalty"
            value={editingProfile?.parameters?.repeat_penalty || ''}
            onChange={e => setEditingProfile({ ...editingProfile, parameters: { ...editingProfile.parameters, repeat_penalty: Number(e.target.value) } })}
            fullWidth margin="normal"
            type="number"
            helperText="Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)"
          />
          <TextField
            label="Temperature"
            value={editingProfile?.parameters?.temperature || ''}
            onChange={e => setEditingProfile({ ...editingProfile, parameters: { ...editingProfile.parameters, temperature: Number(e.target.value) } })}
            fullWidth margin="normal"
            type="number"
            helperText="The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)"
          />
          <TextField
            label="Seed"
            value={editingProfile?.parameters?.seed || ''}
            onChange={e => setEditingProfile({ ...editingProfile, parameters: { ...editingProfile.parameters, seed: Number(e.target.value) } })}
            fullWidth margin="normal"
            type="number"
            helperText="Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt. (Default: 0)"
          />
          <TextField
            label="Stop"
            value={editingProfile?.parameters?.stop || ''}
            onChange={e => setEditingProfile({ ...editingProfile, parameters: { ...editingProfile.parameters, stop: [e.target.value] } })}
            fullWidth margin="normal"
            multiline
            minRows={2}
            helperText="Sets the stop sequences to use. When this pattern is encountered the LLM will stop generating text and return. Multiple stop patterns may be set by specifying multiple separate stop parameters in a modelfile."
          />
          <TextField
            label="Number of Predictions"
            value={editingProfile?.parameters?.num_predict || ''}
            onChange={e => setEditingProfile({ ...editingProfile, parameters: { ...editingProfile.parameters, num_predict: Number(e.target.value) } })}
            fullWidth margin="normal"
            type="number"
            helperText="Maximum number of tokens to predict when generating text. (Default: -1, infinite generation)"
          />
          <TextField
            label="Max Tokens"
            value={editingProfile?.parameters?.max_tokens || ''}
            onChange={e => setEditingProfile({ ...editingProfile, parameters: { ...editingProfile.parameters, max_tokens: Number(e.target.value) } })}
            fullWidth margin="normal"
            type="number"
            helperText="Maximum number of tokens to generate in a single response. This is a hard limit that stops generation."
          />
          <TextField
            label="Batch Size"
            value={editingProfile?.parameters?.batch_size || ''}
            onChange={e => setEditingProfile({ ...editingProfile, parameters: { ...editingProfile.parameters, batch_size: Number(e.target.value) } })}
            fullWidth margin="normal"
            type="number"
            helperText="Batch size for processing inputs. Higher values may improve throughput but use more memory. (Default: depends on model)"
          />
          <TextField
            label="Top K"
            value={editingProfile?.parameters?.top_k || ''}
            onChange={e => setEditingProfile({ ...editingProfile, parameters: { ...editingProfile.parameters, top_k: Number(e.target.value) } })}
            fullWidth margin="normal"
            type="number"
            helperText="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)"
          />
          <TextField
            label="Top P"
            value={editingProfile?.parameters?.top_p || ''}
            onChange={e => setEditingProfile({ ...editingProfile, parameters: { ...editingProfile.parameters, top_p: Number(e.target.value) } })}
            fullWidth margin="normal"
            type="number"
            helperText="Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)"
          />
          <TextField
            label="Minimum Probability"
            value={editingProfile?.parameters?.min_p || ''}
            onChange={e => setEditingProfile({ ...editingProfile, parameters: { ...editingProfile.parameters, min_p: Number(e.target.value) } })}
            fullWidth margin="normal"
            type="number"
            helperText="Alternative to the top_p, and aims to ensure a balance of quality and variety. The parameter p represents the minimum probability for a token to be considered, relative to the probability of the most likely token. For example, with p=0.05 and the most likely token having a probability of 0.9, logits with a value less than 0.045 are filtered out. (Default: 0.0)"
          />

          {/* Circuit Breaker Configuration Section */}
          <Accordion sx={{ mt: 2 }}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <SettingsIcon sx={{ mr: 1 }} />
                <Typography variant="h6">
                  Circuit Breaker Configuration (Optional)
                </Typography>
                {editingProfile?.circuit_breaker && (
                  <Chip
                    label="Custom Settings Active"
                    color="primary"
                    size="small"
                    sx={{ ml: 2 }}
                  />
                )}
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Configure timeout protection, retry behavior, and quality monitoring overrides for this specific model profile.
              </Typography>

              <FormControlLabel
                control={
                  <Checkbox
                    checked={!!editingProfile?.circuit_breaker}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setEditingProfile({
                          ...editingProfile,
                          circuit_breaker: {}
                        });
                      } else {
                        const { circuit_breaker, ...restProfile } = editingProfile!;
                        setEditingProfile(restProfile);
                      }
                    }}
                  />
                }
                label="Override Global Circuit Breaker Settings"
              />

              {editingProfile?.circuit_breaker && (
                <>
                  {/* Timeout Settings */}
                  <Typography variant="subtitle1" sx={{ mt: 2, mb: 1, fontWeight: 'bold' }}>
                    Timeout Settings
                  </Typography>

                  <TextField
                    label="Base Timeout (seconds)"
                    type="number"
                    value={editingProfile?.circuit_breaker?.base_timeout ?? ''}
                    onChange={(e) => {
                      const currentConfig = editingProfile?.circuit_breaker;
                      if (currentConfig !== undefined) {
                        const value = e.target.value === '' ? undefined : Number(e.target.value);
                        setEditingProfile({
                          ...editingProfile,
                          circuit_breaker: {
                            ...currentConfig,
                            base_timeout: value
                          }
                        });
                      }
                    }}
                    fullWidth margin="normal"
                    inputProps={{ min: 1, max: 600, step: 1 }}
                    helperText="Base timeout for model operations (1-600 seconds)"
                  />

                  <TextField
                    label="Deep Research Timeout (seconds)"
                    type="number"
                    value={editingProfile?.circuit_breaker?.deep_research_timeout ?? ''}
                    onChange={(e) => {
                      const currentConfig = editingProfile?.circuit_breaker;
                      if (currentConfig !== undefined) {
                        const value = e.target.value === '' ? undefined : Number(e.target.value);
                        setEditingProfile({
                          ...editingProfile,
                          circuit_breaker: {
                            ...currentConfig,
                            deep_research_timeout: value
                          }
                        });
                      }
                    }}
                    fullWidth margin="normal"
                    inputProps={{ min: 1, max: 1200, step: 1 }}
                    helperText="Extended timeout for research tasks (1-1200 seconds)"
                  />

                  {/* Retry Settings */}
                  <Typography variant="subtitle1" sx={{ mt: 3, mb: 1, fontWeight: 'bold' }}>
                    Retry Settings
                  </Typography>

                  <TextField
                    label="Max Retries"
                    type="number"
                    value={editingProfile?.circuit_breaker?.max_retries ?? ''}
                    onChange={(e) => {
                      const currentConfig = editingProfile?.circuit_breaker;
                      if (currentConfig !== undefined) {
                        const value = e.target.value === '' ? undefined : Number(e.target.value);
                        setEditingProfile({
                          ...editingProfile,
                          circuit_breaker: {
                            ...currentConfig,
                            max_retries: value
                          }
                        });
                      }
                    }}
                    fullWidth margin="normal"
                    inputProps={{ min: 0, max: 10, step: 1 }}
                    helperText="Maximum number of retries before giving up (0-10)"
                  />

                  <TextField
                    label="Cooldown Period (seconds)"
                    type="number"
                    value={editingProfile?.circuit_breaker?.cooldown_period ?? ''}
                    onChange={(e) => {
                      const currentConfig = editingProfile?.circuit_breaker;
                      if (currentConfig !== undefined) {
                        const value = e.target.value === '' ? undefined : Number(e.target.value);
                        setEditingProfile({
                          ...editingProfile,
                          circuit_breaker: {
                            ...currentConfig,
                            cooldown_period: value
                          }
                        });
                      }
                    }}
                    fullWidth margin="normal"
                    inputProps={{ min: 0, max: 300, step: 1 }}
                    helperText="Time before allowing retry after failure (0-300 seconds)"
                  />

                  {/* Quality Monitoring */}
                  <Typography variant="subtitle1" sx={{ mt: 3, mb: 1, fontWeight: 'bold' }}>
                    Quality Monitoring
                  </Typography>

                  <FormControlLabel
                    control={
                      <Switch
                        checked={editingProfile?.circuit_breaker?.enable_perplexity_guard ?? true}
                        onChange={(e) => {
                          const currentConfig = editingProfile?.circuit_breaker;
                          if (currentConfig !== undefined) {
                            setEditingProfile({
                              ...editingProfile,
                              circuit_breaker: {
                                ...currentConfig,
                                enable_perplexity_guard: e.target.checked
                              }
                            });
                          }
                        }}
                      />
                    }
                    label="Enable Perplexity Guard"
                    sx={{ mb: 1, display: 'block' }}
                  />

                  <TextField
                    label="Perplexity Window"
                    type="number"
                    value={editingProfile?.circuit_breaker?.perplexity_window ?? ''}
                    onChange={(e) => {
                      const currentConfig = editingProfile?.circuit_breaker;
                      if (currentConfig !== undefined) {
                        const value = e.target.value === '' ? undefined : Number(e.target.value);
                        setEditingProfile({
                          ...editingProfile,
                          circuit_breaker: {
                            ...currentConfig,
                            perplexity_window: value
                          }
                        });
                      }
                    }}
                    fullWidth margin="normal"
                    inputProps={{ min: 10, max: 200, step: 1 }}
                    helperText="Number of tokens for perplexity calculation (10-200)"
                  />

                  <TextField
                    label="Perplexity Threshold"
                    type="number"
                    value={editingProfile?.circuit_breaker?.perplexity_threshold ?? ''}
                    onChange={(e) => {
                      const currentConfig = editingProfile?.circuit_breaker;
                      if (currentConfig !== undefined) {
                        const value = e.target.value === '' ? undefined : Number(e.target.value);
                        setEditingProfile({
                          ...editingProfile,
                          circuit_breaker: {
                            ...currentConfig,
                            perplexity_threshold: value
                          }
                        });
                      }
                    }}
                    fullWidth margin="normal"
                    inputProps={{ min: 1, max: 50, step: 0.1 }}
                    helperText="Perplexity threshold for quality concerns (1-50)"
                  />

                  <TextField
                    label="Average Log Probability Floor"
                    type="number"
                    value={editingProfile?.circuit_breaker?.avg_logprob_floor ?? ''}
                    onChange={(e) => {
                      const currentConfig = editingProfile?.circuit_breaker;
                      if (currentConfig !== undefined) {
                        const value = e.target.value === '' ? undefined : Number(e.target.value);
                        setEditingProfile({
                          ...editingProfile,
                          circuit_breaker: {
                            ...currentConfig,
                            avg_logprob_floor: value
                          }
                        });
                      }
                    }}
                    fullWidth margin="normal"
                    inputProps={{ min: -20, max: 0, step: 0.1 }}
                    helperText="Minimum average log probability threshold (-20 to 0)"
                  />

                  {/* Repetition Detection */}
                  <Typography variant="subtitle1" sx={{ mt: 3, mb: 1, fontWeight: 'bold' }}>
                    Repetition Detection
                  </Typography>

                  <TextField
                    label="Repetition N-gram Size"
                    type="number"
                    value={editingProfile?.circuit_breaker?.repetition_ngram ?? ''}
                    onChange={(e) => {
                      const currentConfig = editingProfile?.circuit_breaker;
                      if (currentConfig !== undefined) {
                        const value = e.target.value === '' ? undefined : Number(e.target.value);
                        setEditingProfile({
                          ...editingProfile,
                          circuit_breaker: {
                            ...currentConfig,
                            repetition_ngram: value
                          }
                        });
                      }
                    }}
                    fullWidth margin="normal"
                    inputProps={{ min: 2, max: 20, step: 1 }}
                    helperText="N-gram size for repetition detection (2-20)"
                  />

                  <TextField
                    label="Repetition Threshold"
                    type="number"
                    value={editingProfile?.circuit_breaker?.repetition_threshold ?? ''}
                    onChange={(e) => {
                      const currentConfig = editingProfile?.circuit_breaker;
                      if (currentConfig !== undefined) {
                        const value = e.target.value === '' ? undefined : Number(e.target.value);
                        setEditingProfile({
                          ...editingProfile,
                          circuit_breaker: {
                            ...currentConfig,
                            repetition_threshold: value
                          }
                        });
                      }
                    }}
                    fullWidth margin="normal"
                    inputProps={{ min: 2, max: 20, step: 1 }}
                    helperText="Number of repetitions before triggering detection (2-20)"
                  />

                  <TextField
                    label="Tool Generation Repetition N-gram"
                    type="number"
                    value={editingProfile?.circuit_breaker?.tool_gen_repetition_ngram ?? ''}
                    onChange={(e) => {
                      const currentConfig = editingProfile?.circuit_breaker;
                      if (currentConfig !== undefined) {
                        const value = e.target.value === '' ? undefined : Number(e.target.value);
                        setEditingProfile({
                          ...editingProfile,
                          circuit_breaker: {
                            ...currentConfig,
                            tool_gen_repetition_ngram: value
                          }
                        });
                      }
                    }}
                    fullWidth margin="normal"
                    inputProps={{ min: 2, max: 20, step: 1 }}
                    helperText="N-gram size for tool generation repetition detection (2-20)"
                  />

                  <TextField
                    label="Tool Generation Repetition Threshold"
                    type="number"
                    value={editingProfile?.circuit_breaker?.tool_gen_repetition_threshold ?? ''}
                    onChange={(e) => {
                      const currentConfig = editingProfile?.circuit_breaker;
                      if (currentConfig !== undefined) {
                        const value = e.target.value === '' ? undefined : Number(e.target.value);
                        setEditingProfile({
                          ...editingProfile,
                          circuit_breaker: {
                            ...currentConfig,
                            tool_gen_repetition_threshold: value
                          }
                        });
                      }
                    }}
                    fullWidth margin="normal"
                    inputProps={{ min: 2, max: 20, step: 1 }}
                    helperText="Repetitions before triggering tool generation detection (2-20)"
                  />

                  {/* Advanced Settings */}
                  <Typography variant="subtitle1" sx={{ mt: 3, mb: 1, fontWeight: 'bold' }}>
                    Advanced Settings
                  </Typography>

                  <TextField
                    label="Min Tokens for Evaluation"
                    type="number"
                    value={editingProfile?.circuit_breaker?.min_tokens_for_eval ?? ''}
                    onChange={(e) => {
                      const currentConfig = editingProfile?.circuit_breaker;
                      if (currentConfig !== undefined) {
                        const value = e.target.value === '' ? undefined : Number(e.target.value);
                        setEditingProfile({
                          ...editingProfile,
                          circuit_breaker: {
                            ...currentConfig,
                            min_tokens_for_eval: value
                          }
                        });
                      }
                    }}
                    fullWidth margin="normal"
                    inputProps={{ min: 5, max: 500, step: 1 }}
                    helperText="Minimum tokens before starting quality evaluation (5-500)"
                  />

                  <TextField
                    label="Perplexity Log Interval (tokens)"
                    type="number"
                    value={editingProfile?.circuit_breaker?.perplexity_log_interval_tokens ?? ''}
                    onChange={(e) => {
                      const currentConfig = editingProfile?.circuit_breaker;
                      if (currentConfig !== undefined) {
                        const value = e.target.value === '' ? undefined : Number(e.target.value);
                        setEditingProfile({
                          ...editingProfile,
                          circuit_breaker: {
                            ...currentConfig,
                            perplexity_log_interval_tokens: value
                          }
                        });
                      }
                    }}
                    fullWidth margin="normal"
                    inputProps={{ min: 5, max: 100, step: 1 }}
                    helperText="Interval for logging perplexity metrics (5-100 tokens)"
                  />

                  <FormControlLabel
                    control={
                      <Switch
                        checked={editingProfile?.circuit_breaker?.log_repetition_events ?? true}
                        onChange={(e) => {
                          const currentConfig = editingProfile?.circuit_breaker;
                          if (currentConfig !== undefined) {
                            setEditingProfile({
                              ...editingProfile,
                              circuit_breaker: {
                                ...currentConfig,
                                log_repetition_events: e.target.checked
                              }
                            });
                          }
                        }}
                      />
                    }
                    label="Log Repetition Detection Events"
                    sx={{ mt: 1, display: 'block' }}
                  />
                </>
              )}
            </AccordionDetails>
          </Accordion>

          {/* GPU Configuration Section */}
          <Accordion sx={{ mt: 2 }}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <MemoryIcon sx={{ mr: 1 }} />
                <Typography variant="h6">
                  GPU Configuration (Optional)
                </Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Alert severity="info" sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <WarningIcon sx={{ mr: 1 }} />
                  <Typography variant="body2">
                    GPU configuration only applies to local models (llama.cpp, etc.). Remote API models ignore these settings.
                  </Typography>
                </Box>
              </Alert>

              <FormControlLabel
                control={
                  <Checkbox
                    checked={editingProfile?.gpu_config !== undefined}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setEditingProfile({
                          ...editingProfile,
                          gpu_config: {
                            no_kv_offload: false,
                            main_gpu: -1,
                            tensor_split: [],
                            split_mode: 'none',
                            offload_kqv: true
                          }
                        });
                      } else {
                        const { gpu_config, ...restProfile } = editingProfile!;
                        setEditingProfile(restProfile);
                      }
                    }}
                  />
                }
                label="Override Global GPU Settings"
              />

              {editingProfile?.gpu_config && (
                <>
                  {/* Memory Management */}
                  <Typography variant="subtitle1" sx={{ mt: 2, mb: 1, fontWeight: 'bold' }}>
                    Memory Management
                  </Typography>

                  <FormControlLabel
                    control={
                      <Switch
                        checked={editingProfile?.gpu_config?.no_kv_offload ?? false}
                        onChange={(e) => {
                          const gpuConfig = editingProfile?.gpu_config;
                          if (gpuConfig) {
                            setEditingProfile({
                              ...editingProfile,
                              gpu_config: {
                                ...gpuConfig,
                                no_kv_offload: e.target.checked
                              }
                            });
                          }
                        }}
                      />
                    }
                    label="Force KV Cache to CPU (saves VRAM)"
                    sx={{ mb: 1, display: 'block' }}
                  />

                  {/* Device Selection */}
                  <Typography variant="subtitle1" sx={{ mt: 3, mb: 1, fontWeight: 'bold' }}>
                    Device Selection
                  </Typography>

                  <TextField
                    label="Main GPU Device ID"
                    value={editingProfile?.gpu_config?.main_gpu_device_id ?? ''}
                    onChange={(e) => {
                      const gpuConfig = editingProfile?.gpu_config;
                      if (gpuConfig) {
                        const value = e.target.value === '' ? undefined : e.target.value;
                        setEditingProfile({
                          ...editingProfile,
                          gpu_config: {
                            ...gpuConfig,
                            main_gpu_device_id: value
                          }
                        });
                      }
                    }}
                    fullWidth margin="normal"
                    helperText="GPU device ID/name (e.g., 'NVIDIA GeForce RTX 4090', empty = auto-select)"
                  />

                  <TextField
                    label="Main GPU Index"
                    type="number"
                    value={editingProfile?.gpu_config?.main_gpu ?? ''}
                    onChange={(e) => {
                      const gpuConfig = editingProfile?.gpu_config;
                      if (gpuConfig) {
                        const value = e.target.value === '' ? undefined : Number(e.target.value);
                        setEditingProfile({
                          ...editingProfile,
                          gpu_config: {
                            ...gpuConfig,
                            main_gpu: value
                          }
                        });
                      }
                    }}
                    fullWidth margin="normal"
                    helperText="GPU device index (-1 for auto-selection, overridden by device ID above)"
                    inputProps={{ min: -1 }}
                  />

                  <FormControl fullWidth margin="normal">
                    <InputLabel>Model Split Mode</InputLabel>
                    <Select
                      value={editingProfile?.gpu_config?.split_mode || 'none'}
                      onChange={(e) => {
                        const gpuConfig = editingProfile?.gpu_config;
                        if (gpuConfig) {
                          setEditingProfile({
                            ...editingProfile,
                            gpu_config: {
                              ...gpuConfig,
                              split_mode: e.target.value as GPUConfig['split_mode']
                            }
                          });
                        }
                      }}
                      label="Model Split Mode"
                    >
                      <MenuItem value="none">None - Single device</MenuItem>
                      <MenuItem value="layer">Layer - Split by layers</MenuItem>
                      <MenuItem value="row">Row - Split by tensor rows</MenuItem>
                    </Select>
                  </FormControl>

                  {/* Tensor Split Configuration */}
                  <Typography variant="subtitle1" sx={{ mt: 3, mb: 1, fontWeight: 'bold' }}>
                    Tensor Split Configuration
                    <Button
                      onClick={() => {
                        const gpuConfig = editingProfile?.gpu_config;
                        if (gpuConfig) {
                          const newTensorSplit = [...(gpuConfig.tensor_split || []), 0.5];
                          setEditingProfile({
                            ...editingProfile,
                            gpu_config: {
                              ...gpuConfig,
                              tensor_split: newTensorSplit
                            }
                          });
                        }
                      }}
                      startIcon={<AddIcon />}
                      size="small"
                      sx={{ ml: 2 }}
                    >
                      Add Device
                    </Button>
                  </Typography>

                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    Distribute model computation across multiple GPUs. Values must sum to 1.0.
                  </Typography>

                  {editingProfile?.gpu_config?.tensor_split &&
                    editingProfile.gpu_config.tensor_split.length > 0 &&
                    (
                      <>
                        <Box sx={{ mb: 2 }}>
                          {(() => {
                            const tensorSplit = editingProfile?.gpu_config?.tensor_split || [];
                            const sum = tensorSplit.reduce((acc, val) => acc + val, 0);
                            const isValid = Math.abs(sum - 1.0) < 0.01;
                            return (
                              <Typography variant="body2" color={isValid ? 'success.main' : 'error.main'}>
                                Current sum: {sum.toFixed(3)} {isValid ? '✓' : '(must equal 1.0)'}
                              </Typography>
                            );
                          })()}
                        </Box>

                        {editingProfile.gpu_config.tensor_split.map((split, index) => (
                          <Box key={index} sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                            <Typography sx={{ minWidth: 80 }}>
                              Device {index}:
                            </Typography>
                            <Slider
                              value={split}
                              onChange={(_, value) => {
                                const gpuConfig = editingProfile?.gpu_config;
                                if (gpuConfig && gpuConfig.tensor_split) {
                                  const newTensorSplit = [...gpuConfig.tensor_split];
                                  newTensorSplit[index] = value as number;
                                  setEditingProfile({
                                    ...editingProfile,
                                    gpu_config: {
                                      ...gpuConfig,
                                      tensor_split: newTensorSplit
                                    }
                                  });
                                }
                              }}
                              min={0}
                              max={1}
                              step={0.01}
                              sx={{ mx: 2, flex: 1 }}
                              valueLabelDisplay="auto"
                              valueLabelFormat={(value) => value.toFixed(2)}
                            />
                            <Typography sx={{ minWidth: 60, textAlign: 'center' }}>
                              {split.toFixed(2)}
                            </Typography>
                            <IconButton
                              onClick={() => {
                                const gpuConfig = editingProfile?.gpu_config;
                                if (gpuConfig && gpuConfig.tensor_split) {
                                  const newTensorSplit = [...gpuConfig.tensor_split];
                                  newTensorSplit.splice(index, 1);
                                  setEditingProfile({
                                    ...editingProfile,
                                    gpu_config: {
                                      ...gpuConfig,
                                      tensor_split: newTensorSplit
                                    }
                                  });
                                }
                              }}
                              size="small"
                              color="error"
                            >
                              <DeleteIcon />
                            </IconButton>
                          </Box>
                        ))}
                      </>
                    )
                  }

                  {(!editingProfile?.gpu_config?.tensor_split ||
                    editingProfile.gpu_config.tensor_split.length === 0) &&
                    (
                      <Typography variant="body2" color="text.secondary">
                        No tensor split configured. Model will run on a single device.
                      </Typography>
                    )}
                </>
              )}
            </AccordionDetails>
          </Accordion>          {editingProfile?.circuit_breaker && (
            <Button
              variant="outlined"
              color="secondary"
              sx={{ mt: 2 }}
              onClick={() => setEditingProfile({ ...editingProfile, circuit_breaker: undefined })}
            >
              Clear Circuit Breaker Config (Use Global Settings)
            </Button>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>Cancel</Button>
          <Button onClick={() => handleSaveProfile()} variant="contained">Save</Button>
          <Button onClick={() => handleSaveProfile(true)} variant="contained">Save As</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ModelProfilesPage;