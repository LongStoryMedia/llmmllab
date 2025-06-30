import { useEffect, useState } from 'react';
import { Box, Typography, Button, Dialog, DialogTitle, DialogContent, DialogActions, TextField, Paper, IconButton, Grid } from '@mui/material';
import { listModelProfiles, createModelProfile, updateModelProfile, deleteModelProfile } from '../api/model';
import DeleteIcon from '@mui/icons-material/Delete';
import EditIcon from '@mui/icons-material/Edit';
import { ModelProfile } from '../types/ModelProfile';
import { useAuth } from '../auth';
import ModelSelector from '../components/ModelSelector/ModelSelector';
import { getToken } from '../api';
import { ModelProfileType } from '../types/ModelProfileType';

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
              </Box>
              <Box>
                <IconButton onClick={() => {
                  setEditingProfile(profile); setDialogOpen(true); 
                }}><EditIcon /></IconButton>
                <IconButton onClick={() => handleDeleteProfile(profile.id)}><DeleteIcon /></IconButton>
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