import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useState } from 'react';
import { Box, FormControl, InputLabel, Select, MenuItem, Typography, Alert, Snackbar } from '@mui/material';
import { useAuth } from '../../auth';
import { getModels } from '../../api/model';
import { getHeaders, req } from '../../api/base';
import { getToken } from '../../api';
import ControlLoader from '../Shared/ControlLoader';
const ImageModelSelector = ({ onModelChange, mode }) => {
    const [models, setModels] = useState([]);
    const [selectedModel, setSelectedModel] = useState('');
    const [isLoading, setIsLoading] = useState(true);
    const [snackbar, setSnackbar] = useState({
        open: false,
        message: '',
        severity: 'success'
    });
    const auth = useAuth();
    // Load models from API
    useEffect(() => {
        const fetchModels = async () => {
            try {
                setIsLoading(true);
                const allModels = await getModels(getToken(auth.user));
                // Filter models to only include those with TextToImage specialization
                const imageModels = allModels.filter(model => model.details?.specialization === mode);
                setModels(imageModels);
                // If we have models, try to find the active one if any
                // This could be enhanced if the API provides this information in the future
            }
            catch (error) {
                console.error('Failed to fetch image models:', error);
                setSnackbar({
                    open: true,
                    message: 'Failed to load image models',
                    severity: 'error'
                });
            }
            finally {
                setIsLoading(false);
            }
        };
        fetchModels();
    }, [auth.user, mode]);
    const handleModelChange = async (event) => {
        const modelId = event.target.value;
        setSelectedModel(modelId);
        try {
            // Make API request to set the active image model
            await req({
                method: 'PUT',
                path: `api/models/image/${modelId}`,
                headers: getHeaders(getToken(auth.user))
            });
            setSnackbar({
                open: true,
                message: 'Active image model updated successfully',
                severity: 'success'
            });
            if (onModelChange) {
                onModelChange(modelId);
            }
        }
        catch (error) {
            console.error('Failed to update active image model:', error);
            setSnackbar({
                open: true,
                message: 'Failed to update active image model',
                severity: 'error'
            });
        }
    };
    const handleCloseSnackbar = () => {
        setSnackbar(prev => ({ ...prev, open: false }));
    };
    if (isLoading) {
        return _jsx(ControlLoader, { text: 'Loading image models...' });
    }
    return (_jsxs(Box, { sx: { mb: 2, p: 2 }, children: [_jsx(Typography, { variant: "h6", gutterBottom: true, children: "Select Image Generation Model" }), models.length === 0 ? (_jsx(Alert, { severity: "info", children: "No text-to-image models available" })) : (_jsxs(FormControl, { fullWidth: true, children: [_jsx(InputLabel, { id: "image-model-select-label", children: "Image Model" }), _jsx(Select, { labelId: "image-model-select-label", id: "image-model-select", value: selectedModel, onChange: handleModelChange, label: "Image Model", children: models.map((model) => (_jsx(MenuItem, { value: model.id ?? model.model, title: model.details.description || '', children: model.name }, model.model))) })] })), _jsx(Snackbar, { open: snackbar.open, autoHideDuration: 6000, onClose: handleCloseSnackbar, anchorOrigin: { vertical: 'bottom', horizontal: 'center' }, children: _jsx(Alert, { onClose: handleCloseSnackbar, severity: snackbar.severity, sx: { width: '100%' }, variant: "filled", children: snackbar.message }) })] }));
};
export default ImageModelSelector;
