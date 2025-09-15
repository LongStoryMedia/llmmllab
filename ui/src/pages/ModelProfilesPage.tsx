import { useEffect, useState } from 'react';
import { Box, Typography, Button, Dialog, DialogTitle, DialogContent, DialogActions, TextField, Paper, IconButton, Grid, FormControl, InputLabel, Select, MenuItem, Chip, FormControlLabel, Checkbox } from '@mui/material';
import { listModelProfiles, createModelProfile, updateModelProfile, deleteModelProfile } from '../api/model';
import DeleteIcon from '@mui/icons-material/Delete';
import EditIcon from '@mui/icons-material/Edit';
import { ModelProfile } from '../types/ModelProfile';
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
  parameters: {},
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
                <Chip
                  label={getModelProfileTypeName(profile.type)}
                  size="small"
                  variant="outlined"
                  sx={{ mt: 1 }}
                />
              </Box>
              <Box>
                <IconButton onClick={() => {
                  setEditingProfile(profile); setDialogOpen(true);
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
          <Typography variant="h6" sx={{ mt: 3, mb: 2 }}>
            Circuit Breaker Configuration (Optional)
            {editingProfile?.circuit_breaker && (
              <Chip
                label="Custom Settings Active"
                color="primary"
                size="small"
                sx={{ ml: 2 }}
              />
            )}
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Configure custom circuit breaker overrides for this profile. Only set values you want to override from global settings.
            {!editingProfile?.circuit_breaker && " Check 'Override Settings' to add custom configuration."}
          </Typography>

          <FormControlLabel
            control={
              <Checkbox
                checked={!!editingProfile?.circuit_breaker}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                  if (e.target.checked && !editingProfile?.circuit_breaker) {
                    // Create an empty circuit breaker config when enabling overrides
                    setEditingProfile({
                      ...editingProfile,
                      circuit_breaker: {}
                    });
                  } else if (!e.target.checked) {
                    // Remove circuit breaker config when disabling overrides
                    setEditingProfile({
                      ...editingProfile,
                      circuit_breaker: undefined
                    });
                  }
                }}
              />
            }
            label="Override Global Circuit Breaker Settings"
          />

          <TextField
            label="Base Timeout (seconds)"
            value={editingProfile?.circuit_breaker?.base_timeout ?? ''}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
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
            type="number"
            inputProps={{ min: 1, max: 600 }}
            helperText="Override base timeout for this profile (1-600 seconds, empty = use global)"
            disabled={!editingProfile?.circuit_breaker}
            placeholder="Use global setting"
          />

          <TextField
            label="Deep Research Timeout (seconds)"
            value={editingProfile?.circuit_breaker?.deep_research_timeout ?? ''}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
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
            type="number"
            inputProps={{ min: 1, max: 1200 }}
            helperText="Override deep research timeout (1-1200 seconds, empty = use global)"
            disabled={!editingProfile?.circuit_breaker}
            placeholder="Use global setting"
          />

          <TextField
            label="Max Retries"
            value={editingProfile?.circuit_breaker?.max_retries ?? ''}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
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
            type="number"
            inputProps={{ min: 0, max: 10 }}
            helperText="Override maximum retry attempts (0-10, empty = use global)"
            disabled={!editingProfile?.circuit_breaker}
            placeholder="Use global setting"
          />

          <TextField
            label="Cooldown Period (seconds)"
            value={editingProfile?.circuit_breaker?.cooldown_period ?? ''}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
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
            type="number"
            inputProps={{ min: 0, max: 300 }}
            helperText="Override cooldown period (0-300 seconds, empty = use global)"
            disabled={!editingProfile?.circuit_breaker}
            placeholder="Use global setting"
          />

          <FormControl fullWidth margin="normal" disabled={!editingProfile?.circuit_breaker}>
            <InputLabel>Perplexity Guard</InputLabel>
            <Select
              value={
                editingProfile?.circuit_breaker?.enable_perplexity_guard === undefined
                  ? 'global'
                  : editingProfile?.circuit_breaker?.enable_perplexity_guard
                    ? 'enabled'
                    : 'disabled'
              }
              onChange={(e) => {
                const currentConfig = editingProfile?.circuit_breaker;
                if (currentConfig !== undefined) {
                  let newValue: boolean | undefined;
                  if (e.target.value === 'global') {
                    newValue = undefined;
                  } else if (e.target.value === 'enabled') {
                    newValue = true;
                  } else {
                    newValue = false;
                  }

                  setEditingProfile({
                    ...editingProfile,
                    circuit_breaker: {
                      ...currentConfig,
                      enable_perplexity_guard: newValue
                    }
                  });
                }
              }}
            >
              <MenuItem value="global">Use Global Setting</MenuItem>
              <MenuItem value="enabled">Enable Perplexity Guard</MenuItem>
              <MenuItem value="disabled">Disable Perplexity Guard</MenuItem>
            </Select>
          </FormControl>

          <TextField
            label="Perplexity Window"
            value={editingProfile?.circuit_breaker?.perplexity_window ?? ''}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
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
            type="number"
            inputProps={{ min: 10, max: 200 }}
            helperText="Override perplexity window (10-200 tokens, empty = use global)"
            disabled={!editingProfile?.circuit_breaker}
            placeholder="Use global setting"
          />

          <TextField
            label="Perplexity Threshold"
            value={editingProfile?.circuit_breaker?.perplexity_threshold ?? ''}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
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
            type="number"
            inputProps={{ min: 1, max: 50, step: 0.1 }}
            helperText="Override perplexity threshold (1-50, empty = use global)"
            disabled={!editingProfile?.circuit_breaker}
            placeholder="Use global setting"
          />

          <TextField
            label="Average Log Probability Floor"
            value={editingProfile?.circuit_breaker?.avg_logprob_floor ?? ''}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
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
            type="number"
            inputProps={{ min: -20, max: 0, step: 0.1 }}
            helperText="Override log probability floor (-20 to 0, empty = use global)"
            disabled={!editingProfile?.circuit_breaker}
            placeholder="Use global setting"
          />

          {editingProfile?.circuit_breaker && (
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