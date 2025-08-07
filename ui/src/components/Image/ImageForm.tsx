import React from 'react';
import { 
  Box, 
  TextField, 
  Button, 
  CardMedia, 
  Typography, 
  Slider,
  Paper,
  Divider
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import EditIcon from '@mui/icons-material/Edit';
import { ImageMetadata } from '../../types/ImageMetadata';
import ImageModelSelector from '../ModelSelector/ImageModelSelector';
import { TooltipToggleButton } from '../Chat/TooltipToggleButton';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import { useConfigContext } from '../../context/ConfigContext';


interface ImageFormProps {
  selectedImage: number | null;
  selectedImageData: ImageMetadata | null;
  prompt: string;
  setPrompt: (prompt: string) => void;
  negativePrompt: string;
  setNegativePrompt: (prompt: string) => void;
  width: number;
  setWidth: (width: number) => void;
  height: number;
  setHeight: (height: number) => void;
  inferenceSteps: number;
  setInferenceSteps: (steps: number) => void;
  guidanceScale: number;
  setGuidanceScale: (scale: number) => void;
  selectedModel: string;
  setSelectedModel: (model: string) => void;
  isGenerating: boolean;
  onGenerateImage: () => void;
  onEditImage: () => void;
  onCancelEdit: () => void;
}

const ImageForm: React.FC<ImageFormProps> = ({
  selectedImage,
  selectedImageData,
  prompt,
  setPrompt,
  negativePrompt,
  setNegativePrompt,
  width,
  setWidth,
  height,
  setHeight,
  inferenceSteps,
  setInferenceSteps,
  guidanceScale,
  setGuidanceScale,
  selectedModel,
  // @ts-expect-error ts()
  setSelectedModel, // eslint-disable-line @typescript-eslint/no-unused-vars
  isGenerating,
  onGenerateImage,
  onEditImage,
  onCancelEdit
}) => {
  const { config, updatePartialConfig } = useConfigContext();
  const [autoPrompt, setAutoPrompt] = React.useState<boolean>(config?.image_generation?.auto_prompt_refinement || false);

  const getImgViewUrl = (image: ImageMetadata) => {
    return `${image.view_url || image.download_url}`;
  };

  const handleAutoRefineToggle = async () => {
    setAutoPrompt(!autoPrompt);
    await updatePartialConfig('image_generation', {
      ...config!.image_generation!,
      auto_prompt_refinement: !autoPrompt
    });
  }

  return (
    <Paper elevation={3} sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        {selectedImage ? 'Edit Image' : 'Generate New Image'}
      </Typography>
      <Divider sx={{ mb: 2 }} />
      
      {selectedImage && selectedImageData && (
        <Box sx={{ mb: 2, textAlign: 'center' }}>
          <CardMedia
            component="img"
            sx={{ 
              maxHeight: '400px', 
              width: 'auto', 
              maxWidth: '100%', 
              margin: '0 auto',
              objectFit: 'contain'
            }}
            image={getImgViewUrl(selectedImageData)}
            alt="Selected image"
          />
        </Box>
      )}
      
      {/* Prompt Input */}
      <TextField
        fullWidth
        label="Prompt"
        multiline
        rows={3}
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        margin="normal"
        variant="outlined"
        placeholder="Describe the image you want to generate..."
      />
      
      <TextField
        fullWidth
        label="Negative Prompt"
        multiline
        rows={2}
        value={negativePrompt}
        onChange={(e) => setNegativePrompt(e.target.value)}
        margin="normal"
        variant="outlined"
        placeholder="Describe what you want to avoid in the image..."
      />
      
      {/* Image Settings */}
      <Box sx={{ mt: 1, display: 'flex', flexWrap: 'wrap' }}>
        <Box sx={{ width: { xs: '100%', sm: '50%' }, pr: { xs: 0, sm: 1 }, mb: 2 }}>
          <Typography gutterBottom>Width: {width}px</Typography>
          <Slider
            value={width}
            onChange={(_, value) => setWidth(value as number)}
            min={256}
            max={1536}
            step={64}
            valueLabelDisplay="auto"
          />
        </Box>
        <Box sx={{ width: { xs: '100%', sm: '50%' }, pl: { xs: 0, sm: 1 }, mb: 2 }}>
          <Typography gutterBottom>Height: {height}px</Typography>
          <Slider
            value={height}
            onChange={(_, value) => setHeight(value as number)}
            min={256}
            max={1536}
            step={64}
            valueLabelDisplay="auto"
          />
        </Box>
        <Box sx={{ width: { xs: '100%', sm: '50%' }, pr: { xs: 0, sm: 1 }, mb: 2 }}>
          <Typography gutterBottom>Inference Steps: {inferenceSteps}</Typography>
          <Slider
            value={inferenceSteps}
            onChange={(_, value) => setInferenceSteps(value as number)}
            min={5}
            max={100}
            step={1}
            valueLabelDisplay="auto"
          />
        </Box>
        <Box sx={{ width: { xs: '100%', sm: '50%' }, pl: { xs: 0, sm: 1 }, mb: 2 }}>
          <Typography gutterBottom>Guidance Scale: {guidanceScale.toFixed(1)}</Typography>
          <Slider
            value={guidanceScale}
            onChange={(_, value) => setGuidanceScale(value as number)}
            min={1}
            max={20}
            step={0.1}
            valueLabelDisplay="auto"
          />
        </Box>
      </Box>
      
      <ImageModelSelector mode={selectedModel ? 'ImageToImage' : 'TextToImage'}/>

      <TooltipToggleButton 
        value="autoPromptRefinement"
        aria-label="auto prompt refinement"
        color="secondary"
        tooltip="Automatically refine your image prompts to improve generation quality."
        selected={autoPrompt}
        onSelect={handleAutoRefineToggle}
      >
        <AutoFixHighIcon sx={{ fontSize: 'small' }} />
        <Typography 
          variant="body2" 
        >
          Auto Refine
        </Typography>
      </TooltipToggleButton>
      
      <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
        <Button
          variant="contained"
          color={selectedImage ? "secondary" : "primary"}
          disabled={!prompt || isGenerating}
          onClick={selectedImage ? onEditImage : onGenerateImage}
          startIcon={selectedImage ? <EditIcon /> : <AddIcon />}
          size="large"
          sx={{ minWidth: 200 }}
        >
          {isGenerating 
            ? "Generating..." 
            : selectedImage 
              ? "Edit Image" 
              : "Generate Image"}
        </Button>
        
        {selectedImage && (
          <Button
            variant="outlined"
            sx={{ ml: 2 }}
            onClick={onCancelEdit}
          >
            Cancel
          </Button>
        )}
      </Box>
    </Paper>
  );
};

export default ImageForm;
