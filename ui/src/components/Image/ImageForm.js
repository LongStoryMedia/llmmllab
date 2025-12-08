import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React from 'react';
import { Box, TextField, Button, CardMedia, Typography, Slider, Paper, Divider } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import EditIcon from '@mui/icons-material/Edit';
import ImageModelSelector from '../ModelSelector/ImageModelSelector';
import { TooltipToggleButton } from '../Chat/TooltipToggleButton';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import { useConfigContext } from '../../context/ConfigContext';
const ImageForm = ({ selectedImage, selectedImageData, prompt, setPrompt, negativePrompt, setNegativePrompt, width, setWidth, height, setHeight, inferenceSteps, setInferenceSteps, guidanceScale, setGuidanceScale, selectedModel, 
// @ts-expect-error ts()
setSelectedModel, // eslint-disable-line @typescript-eslint/no-unused-vars
isGenerating, onGenerateImage, onEditImage, onCancelEdit }) => {
    const { config, updatePartialConfig } = useConfigContext();
    const [autoPrompt, setAutoPrompt] = React.useState(config?.image_generation?.auto_prompt_refinement || false);
    const getImgViewUrl = (image) => {
        return `${image.view_url || image.download_url}`;
    };
    const handleAutoRefineToggle = async () => {
        setAutoPrompt(!autoPrompt);
        await updatePartialConfig('image_generation', {
            ...config.image_generation,
            auto_prompt_refinement: !autoPrompt
        });
    };
    return (_jsxs(Paper, { elevation: 3, sx: { p: 2 }, children: [_jsx(Typography, { variant: "h6", gutterBottom: true, children: selectedImage ? 'Edit Image' : 'Generate New Image' }), _jsx(Divider, { sx: { mb: 2 } }), selectedImage && selectedImageData && (_jsx(Box, { sx: { mb: 2, textAlign: 'center' }, children: _jsx(CardMedia, { component: "img", sx: {
                        maxHeight: '400px',
                        width: 'auto',
                        maxWidth: '100%',
                        margin: '0 auto',
                        objectFit: 'contain'
                    }, image: getImgViewUrl(selectedImageData), alt: "Selected image" }) })), _jsx(TextField, { fullWidth: true, label: "Prompt", multiline: true, rows: 3, value: prompt, onChange: (e) => setPrompt(e.target.value), margin: "normal", variant: "outlined", placeholder: "Describe the image you want to generate..." }), _jsx(TextField, { fullWidth: true, label: "Negative Prompt", multiline: true, rows: 2, value: negativePrompt, onChange: (e) => setNegativePrompt(e.target.value), margin: "normal", variant: "outlined", placeholder: "Describe what you want to avoid in the image..." }), _jsxs(Box, { sx: { mt: 1, display: 'flex', flexWrap: 'wrap' }, children: [_jsxs(Box, { sx: { width: { xs: '100%', sm: '50%' }, pr: { xs: 0, sm: 1 }, mb: 2 }, children: [_jsxs(Typography, { gutterBottom: true, children: ["Width: ", width, "px"] }), _jsx(Slider, { value: width, onChange: (_, value) => setWidth(value), min: 256, max: 1536, step: 64, valueLabelDisplay: "auto" })] }), _jsxs(Box, { sx: { width: { xs: '100%', sm: '50%' }, pl: { xs: 0, sm: 1 }, mb: 2 }, children: [_jsxs(Typography, { gutterBottom: true, children: ["Height: ", height, "px"] }), _jsx(Slider, { value: height, onChange: (_, value) => setHeight(value), min: 256, max: 1536, step: 64, valueLabelDisplay: "auto" })] }), _jsxs(Box, { sx: { width: { xs: '100%', sm: '50%' }, pr: { xs: 0, sm: 1 }, mb: 2 }, children: [_jsxs(Typography, { gutterBottom: true, children: ["Inference Steps: ", inferenceSteps] }), _jsx(Slider, { value: inferenceSteps, onChange: (_, value) => setInferenceSteps(value), min: 5, max: 100, step: 1, valueLabelDisplay: "auto" })] }), _jsxs(Box, { sx: { width: { xs: '100%', sm: '50%' }, pl: { xs: 0, sm: 1 }, mb: 2 }, children: [_jsxs(Typography, { gutterBottom: true, children: ["Guidance Scale: ", guidanceScale.toFixed(1)] }), _jsx(Slider, { value: guidanceScale, onChange: (_, value) => setGuidanceScale(value), min: 1, max: 20, step: 0.1, valueLabelDisplay: "auto" })] })] }), _jsx(ImageModelSelector, { mode: selectedModel ? 'ImageToImage' : 'TextToImage' }), _jsxs(TooltipToggleButton, { value: "autoPromptRefinement", "aria-label": "auto prompt refinement", color: "secondary", tooltip: "Automatically refine your image prompts to improve generation quality.", selected: autoPrompt, onSelect: handleAutoRefineToggle, children: [_jsx(AutoFixHighIcon, { sx: { fontSize: 'small' } }), _jsx(Typography, { variant: "body2", children: "Auto Refine" })] }), _jsxs(Box, { sx: { mt: 2, display: 'flex', justifyContent: 'center' }, children: [_jsx(Button, { variant: "contained", color: selectedImage ? "secondary" : "primary", disabled: !prompt || isGenerating, onClick: selectedImage ? onEditImage : onGenerateImage, startIcon: selectedImage ? _jsx(EditIcon, {}) : _jsx(AddIcon, {}), size: "large", sx: { minWidth: 200 }, children: isGenerating
                            ? "Generating..."
                            : selectedImage
                                ? "Edit Image"
                                : "Generate Image" }), selectedImage && (_jsx(Button, { variant: "outlined", sx: { ml: 2 }, onClick: onCancelEdit, children: "Cancel" }))] })] }));
};
export default ImageForm;
