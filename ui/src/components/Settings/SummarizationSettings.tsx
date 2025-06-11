import { useState, useEffect } from 'react';
import { Box, TextField, Typography, Button, Switch, FormControlLabel, Alert, Slider } from '@mui/material';
import { useConfigContext } from '../../context/ConfigContext';

const SummarizationSettings = () => {
  const { config, updatePartialConfig, isLoading } = useConfigContext();
  const [localConfig, setLocalConfig] = useState({
    enabled: true,
    messagesBeforeSummary: 10,
    summariesBeforeConsolidation: 5,
    embeddingDimension: 768,
    maxSummaryLevels: 3,
    summaryWeightCoefficient: 0.7
  });
  
  const [saveStatus, setSaveStatus] = useState<{success?: boolean; message: string} | null>(null);

  // Update local state when the user config loads or changes
  useEffect(() => {
    if (config?.summarization) {
      setLocalConfig({
        enabled: config.summarization.enabled !== false,
        messagesBeforeSummary: config.summarization.messages_before_summary || 10,
        summariesBeforeConsolidation: config.summarization.summaries_before_consolidation || 5,
        embeddingDimension: config.summarization.embedding_dimension || 768,
        maxSummaryLevels: config.summarization.max_summary_levels || 3,
        summaryWeightCoefficient: config.summarization.summary_weight_coefficient || 0.7
      });
    }
  }, [config]);

  const handleToggleEnabled = () => {
    setLocalConfig({
      ...localConfig,
      enabled: !localConfig.enabled
    });
  };

  const handleWeightChange = (_event: Event, newValue: number | number[]) => {
    setLocalConfig({
      ...localConfig,
      summaryWeightCoefficient: newValue as number
    });
  };

  const handleSave = async () => {
    setSaveStatus(null);
    try {
      const success = await updatePartialConfig('summarization', localConfig);
      
      if (success) {
        setSaveStatus({
          success: true,
          message: 'Summarization settings saved successfully!'
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
    }
  };

  if (isLoading) {
    return <Box sx={{ padding: 2 }}><Typography>Loading summarization settings...</Typography></Box>;
  }

  return (
    <Box sx={{ padding: 2 }}>
      <Typography variant="h6" gutterBottom>
        Conversation Summarization Settings
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
      
      <FormControlLabel
        control={
          <Switch 
            checked={localConfig.enabled} 
            onChange={handleToggleEnabled} 
          />
        }
        label="Enable Conversation Summarization"
        sx={{ mb: 2, display: 'block' }}
      />
      
      {localConfig.enabled && (
        <>
          <TextField
            label="Messages Before Summary"
            type="number"
            value={localConfig.messagesBeforeSummary}
            onChange={(e) => setLocalConfig({...localConfig, messagesBeforeSummary: parseInt(e.target.value) || 10})}
            fullWidth
            margin="normal"
            helperText="Number of messages before generating a summary"
          />
          
          <TextField
            label="Summaries Before Consolidation"
            type="number"
            value={localConfig.summariesBeforeConsolidation}
            onChange={(e) => setLocalConfig({...localConfig, summariesBeforeConsolidation: parseInt(e.target.value) || 5})}
            fullWidth
            margin="normal"
            helperText="Number of summaries before consolidating them"
          />
          
          <TextField
            label="Embedding Dimension"
            type="number"
            value={localConfig.embeddingDimension}
            onChange={(e) => setLocalConfig({...localConfig, embeddingDimension: parseInt(e.target.value) || 768})}
            fullWidth
            margin="normal"
            helperText="Dimension of the embedding vectors"
          />
          
          <TextField
            label="Max Summary Levels"
            type="number"
            value={localConfig.maxSummaryLevels}
            onChange={(e) => setLocalConfig({...localConfig, maxSummaryLevels: parseInt(e.target.value) || 3})}
            fullWidth
            margin="normal"
            helperText="Maximum depth of summary hierarchy"
          />
          
          <Box sx={{ mt: 3, mb: 2 }}>
            <Typography id="weight-coefficient-slider" gutterBottom>
              Summary Weight Coefficient: {localConfig.summaryWeightCoefficient.toFixed(2)}
            </Typography>
            <Slider
              value={localConfig.summaryWeightCoefficient}
              onChange={handleWeightChange}
              aria-labelledby="weight-coefficient-slider"
              step={0.05}
              marks
              min={0.1}
              max={1.0}
              valueLabelDisplay="auto"
            />
            <Typography variant="body2" color="text.secondary">
              Weight reduction factor for deeper summaries (lower values give less weight to older summaries)
            </Typography>
          </Box>
        </>
      )}
      
      <Button 
        variant="contained" 
        color="primary" 
        sx={{ mt: 2 }} 
        onClick={handleSave}
      >
        Save Summarization Settings
      </Button>
    </Box>
  );
};

export default SummarizationSettings;