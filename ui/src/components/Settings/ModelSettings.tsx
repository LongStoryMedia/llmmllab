import { useState } from 'react';
import { Box, Typography, Button, Alert, Grid } from '@mui/material';
import { useConfigContext } from '../../context/ConfigContext';
import { useAuth } from '../../auth';
import { getToken, updateConfig } from '../../api';
import ModelProfileSelector from '../ModelSelector/ModelProfileSelector';
import { ModelProfileConfig } from '../../types/ModelProfileConfig';

const TASKS: { key: keyof ModelProfileConfig; label: string; }[] = [
  { key: 'primary_profile_id', label: 'Primary' },
  { key: 'summarization_profile_id', label: 'Summary' },
  { key: 'master_summary_profile_id', label: 'Master Summary' },
  { key: 'brief_summary_profile_id', label: 'Brief Summary' },
  { key: 'key_points_profile_id', label: 'Key Points' },
  { key: 'self_critique_profile_id', label: 'Self Critique' },
  { key: 'improvement_profile_id', label: 'Improvement' },
  { key: 'memory_retrieval_profile_id', label: 'Memory Retrieval' },
  { key: 'analysis_profile_id', label: 'Analysis' },
  { key: 'research_task_profile_id', label: 'Research Task' },
  { key: 'research_plan_profile_id', label: 'Research Plan' },
  { key: 'research_consolidation_profile_id', label: 'Research Consolidation' },
  { key: 'research_analysis_profile_id', label: 'Research Analysis' },
  { key: 'embedding_profile_id', label: 'Embeddings' },
  { key: 'formatting_profile_id', label: 'Formatting' },
  { key: 'image_generation_profile_id', label: 'Image Generation' }
];

const ModelSettings = () => {
  const { config, isLoading } = useConfigContext();
  const [saveStatus, setSaveStatus] = useState<{success?: boolean; message: string} | null>(null);
  const auth = useAuth();
  const [isSaving, setIsSaving] = useState(false);

  const handleSave = async () => {
    setSaveStatus(null);
    setIsSaving(true);
    
    try {
      if (!config) {
        setSaveStatus({
          success: false,
          message: 'No configuration available to save.'
        });
        return;
      }

      console.log('Saving config:', config);

      const success = await updateConfig(getToken(auth.user), config)
      
      if (success) {
        setSaveStatus({
          success: true,
          message: 'Model settings saved successfully!'
        });
      } else {
        setSaveStatus({
          success: false,
          message: 'Failed to save settings.'
        });
      }
    } catch (err) {
      setSaveStatus({
        success: false,
        message: `Error: ${err instanceof Error ? err.message : String(err)}`
      });
    } finally {
      setIsSaving(false);
    }
  };

  if (isLoading) {
    return <Box sx={{ padding: 2 }}><Typography>Loading model settings...</Typography></Box>;
  }

  return (
    <Box sx={{ padding: 2 }}>
      <Typography variant="h6" gutterBottom>
        Model Settings
      </Typography>
      
      {saveStatus && (
        <Alert 
          severity={saveStatus.success ? "success" : "error"} 
          sx={{ mb: 2 }}
          onClose={() => setSaveStatus(null)}
        >
          {saveStatus.message}
        </Alert>
      )}
      
      <Box sx={{ mt: 4 }}>
        <Typography variant="h6">Assign Profiles to Tasks</Typography>
        <Grid container spacing={2} sx={{ p: 2 }}>
          {TASKS.map(task => (<ModelProfileSelector key={task.key} task={task} />))}
        </Grid>
      </Box>
      
      <Button 
        variant="contained" 
        color="primary" 
        sx={{ mt: 2 }} 
        onClick={handleSave}
        disabled={isSaving}
      >
        {isSaving ? 'Saving...' : 'Save Model Settings'}
      </Button>
    </Box>
  );
};

export default ModelSettings;