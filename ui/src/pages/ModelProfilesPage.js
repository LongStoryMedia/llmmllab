import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useEffect, useState } from 'react';
import { Box, Typography, Button, Dialog, DialogTitle, DialogContent, DialogActions, TextField, Paper, IconButton, Grid, FormControl, InputLabel, Select, MenuItem, Chip, FormControlLabel, Checkbox, Switch, Slider, Accordion, AccordionSummary, AccordionDetails, Alert } from '@mui/material';
import { Delete as DeleteIcon, Edit as EditIcon, Add as AddIcon, ExpandMore as ExpandMoreIcon, Memory as MemoryIcon, Warning as WarningIcon, Settings as SettingsIcon } from '@mui/icons-material';
import { listModelProfiles, createModelProfile, updateModelProfile, deleteModelProfile } from '../api/model';
import { useAuth } from '../auth';
import ModelSelector from '../components/ModelSelector/ModelSelector';
import { getToken } from '../api';
import { ModelProfileType } from '../types/ModelProfileType';
import { ParameterTuningStrategyValues } from '../types/ParameterTuningStrategy';
import { createDefaultPerformanceParameter, getAllParameterDisplayInfo } from '../utils/parameterUtils';
const getModelProfileTypeName = (type) => {
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
// Helper function to create default parameter optimization config using dynamic system
const createDefaultParameterOptimizationConfig = () => ({
    enabled: true,
    parameters: [
        createDefaultPerformanceParameter('n_ctx'),
        createDefaultPerformanceParameter('n_gpu_layers'),
        createDefaultPerformanceParameter('n_batch')
    ],
    crash_prevention: {
        enable_preallocation_test: true,
        memory_buffer_mb: 1024,
        timeout_seconds: 300,
        enable_graceful_degradation: true
    }
});
const emptyProfile = {
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
    const [profiles, setProfiles] = useState([]);
    const [editingProfile, setEditingProfile] = useState(emptyProfile);
    const [dialogOpen, setDialogOpen] = useState(false);
    const auth = useAuth();
    // Fetch profiles on mount
    useEffect(() => {
        const fetchProfiles = async () => {
            try {
                // You may need to pass the token here
                const data = await listModelProfiles(getToken(auth.user));
                setProfiles(data);
            }
            catch (err) {
                if (err instanceof Error) {
                    console.error('Error fetching model profiles:', err.message);
                }
            }
        };
        fetchProfiles();
    }, [auth.user]);
    // Handle add/edit profile
    const handleSaveProfile = async (isNew = false) => {
        const token = getToken(auth.user);
        if (editingProfile?.id && !isNew) {
            await updateModelProfile(token, editingProfile.id, editingProfile);
        }
        else {
            if (!editingProfile) {
                return;
            }
            // For "Save As" (isNew=true) or new profiles without ID, omit the ID to let backend generate a new one
            let profileToSave = editingProfile;
            if (isNew || !editingProfile.id) {
                const { id, ...profileWithoutId } = editingProfile;
                profileToSave = profileWithoutId;
            }
            await createModelProfile(token, profileToSave);
        }
        setDialogOpen(false);
        setEditingProfile(emptyProfile);
        // Refresh list
        const data = await listModelProfiles(token);
        setProfiles(data);
    };
    // Handle delete
    const handleDeleteProfile = async (id) => {
        const token = getToken(auth.user);
        await deleteModelProfile(token, id);
        setProfiles(profiles.filter(p => p.id !== id));
    };
    return (_jsxs(Box, { sx: { p: 2 }, children: [_jsx(Typography, { variant: "h5", gutterBottom: true, children: "Model Profiles" }), _jsx(Button, { variant: "contained", onClick: () => {
                    setEditingProfile(emptyProfile);
                    setDialogOpen(true);
                }, children: "Add Profile" }), _jsx(Grid, { container: true, spacing: 2, sx: { mt: 2, display: 'flex', flexDirection: 'column' }, children: profiles && profiles.map(profile => (_jsx(Grid, { sx: { p: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }, children: _jsxs(Paper, { sx: { p: 2, textAlign: 'left', width: '100%', display: 'flex', justifyContent: 'space-between' }, children: [_jsxs(Box, { children: [_jsx(Typography, { variant: "subtitle1", children: profile.name }), _jsx(Typography, { variant: "body2", children: profile.description }), _jsxs(Box, { sx: { display: 'flex', gap: 1, mt: 1 }, children: [_jsx(Chip, { label: getModelProfileTypeName(profile.type), size: "small", variant: "outlined", sx: { mt: 1 } }), profile.parameters?.think && (_jsx(Chip, { label: "Think Mode", size: "small", color: "primary", variant: "outlined", sx: { mt: 1 } }))] })] }), _jsxs(Box, { children: [_jsx(IconButton, { onClick: () => {
                                            // Ensure think field is properly set when editing
                                            setEditingProfile({
                                                ...profile,
                                                parameters: {
                                                    ...profile.parameters,
                                                    think: profile.parameters?.think ?? false
                                                }
                                            });
                                            setDialogOpen(true);
                                        }, children: _jsx(EditIcon, {}) }), _jsx(IconButton, { onClick: () => profile.id && handleDeleteProfile(profile.id), children: _jsx(DeleteIcon, {}) })] })] }) }, profile.id))) }), _jsxs(Dialog, { open: dialogOpen, onClose: () => setDialogOpen(false), maxWidth: "sm", fullWidth: true, children: [_jsx(DialogTitle, { children: editingProfile?.id ? 'Edit Profile' : 'Add Profile' }), _jsxs(DialogContent, { children: [_jsx(TextField, { label: "Name", value: editingProfile?.name || '', onChange: e => setEditingProfile({ ...editingProfile, name: e.target.value }), fullWidth: true, margin: "normal" }), _jsx(TextField, { label: "Description", value: editingProfile?.description || '', onChange: e => setEditingProfile({ ...editingProfile, description: e.target.value }), fullWidth: true, margin: "normal" }), _jsxs(FormControl, { fullWidth: true, margin: "normal", children: [_jsx(InputLabel, { children: "Profile Type" }), _jsx(Select, { value: editingProfile?.type ?? ModelProfileType.Primary, onChange: e => setEditingProfile({ ...editingProfile, type: e.target.value }), label: "Profile Type", children: Object.values(ModelProfileType).filter(v => typeof v === 'number').map((type) => (_jsx(MenuItem, { value: type, children: getModelProfileTypeName(type) }, type))) })] }), _jsx(ModelSelector, { onSelect: e => setEditingProfile({ ...editingProfile, model_name: e.target.value }), name: editingProfile?.model_name || '' }), _jsx(ModelSelector, { onSelect: e => setEditingProfile({ ...editingProfile, draft_model: e.target.value }), name: editingProfile?.draft_model || 'draft model', label: "Draft Model (optional)", optional: true }), _jsx(TextField, { label: "System Prompt", value: editingProfile?.system_prompt || '', onChange: e => setEditingProfile({ ...editingProfile, system_prompt: e.target.value }), fullWidth: true, margin: "normal", multiline: true, minRows: 2 }), _jsx(FormControlLabel, { control: _jsx(Switch, { checked: editingProfile?.parameters?.think ?? false, onChange: (e) => setEditingProfile({
                                        ...editingProfile,
                                        parameters: {
                                            ...editingProfile?.parameters,
                                            think: e.target.checked
                                        }
                                    }) }), label: "Enable Think Mode", sx: { mb: 1, display: 'block' } }), _jsx(Typography, { variant: "body2", color: "text.secondary", sx: { mb: 2, mt: -1 }, children: "When enabled, the model will show its internal reasoning process and thoughts before providing the final answer." }), _jsx(TextField, { label: "Number of Context", value: editingProfile?.parameters?.num_ctx || '', onChange: e => setEditingProfile({ ...editingProfile, parameters: { ...editingProfile.parameters, num_ctx: Number(e.target.value) } }), fullWidth: true, margin: "normal", type: "number", helperText: "Sets the size of the context window used to generate the next token. (Default: 2048)" }), _jsx(TextField, { label: "Repeat Last N", value: editingProfile?.parameters?.repeat_last_n || '', onChange: e => setEditingProfile({ ...editingProfile, parameters: { ...editingProfile.parameters, repeat_last_n: Number(e.target.value) } }), fullWidth: true, margin: "normal", type: "number", helperText: "\tSets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)" }), _jsx(TextField, { label: "Repeat Penalty", value: editingProfile?.parameters?.repeat_penalty || '', onChange: e => setEditingProfile({ ...editingProfile, parameters: { ...editingProfile.parameters, repeat_penalty: Number(e.target.value) } }), fullWidth: true, margin: "normal", type: "number", helperText: "Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)" }), _jsx(TextField, { label: "Temperature", value: editingProfile?.parameters?.temperature || '', onChange: e => setEditingProfile({ ...editingProfile, parameters: { ...editingProfile.parameters, temperature: Number(e.target.value) } }), fullWidth: true, margin: "normal", type: "number", helperText: "The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)" }), _jsx(TextField, { label: "Seed", value: editingProfile?.parameters?.seed || '', onChange: e => setEditingProfile({ ...editingProfile, parameters: { ...editingProfile.parameters, seed: Number(e.target.value) } }), fullWidth: true, margin: "normal", type: "number", helperText: "Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt. (Default: 0)" }), _jsx(TextField, { label: "Stop", value: editingProfile?.parameters?.stop || '', onChange: e => setEditingProfile({ ...editingProfile, parameters: { ...editingProfile.parameters, stop: [e.target.value] } }), fullWidth: true, margin: "normal", multiline: true, minRows: 2, helperText: "Sets the stop sequences to use. When this pattern is encountered the LLM will stop generating text and return. Multiple stop patterns may be set by specifying multiple separate stop parameters in a modelfile." }), _jsx(TextField, { label: "Number of Predictions", value: editingProfile?.parameters?.num_predict || '', onChange: e => setEditingProfile({ ...editingProfile, parameters: { ...editingProfile.parameters, num_predict: Number(e.target.value) } }), fullWidth: true, margin: "normal", type: "number", helperText: "Maximum number of tokens to predict when generating text. (Default: -1, infinite generation)" }), _jsx(TextField, { label: "Max Tokens", value: editingProfile?.parameters?.max_tokens || '', onChange: e => setEditingProfile({ ...editingProfile, parameters: { ...editingProfile.parameters, max_tokens: Number(e.target.value) } }), fullWidth: true, margin: "normal", type: "number", helperText: "Maximum number of tokens to generate in a single response. This is a hard limit that stops generation." }), _jsx(TextField, { label: "Batch Size", value: editingProfile?.parameters?.batch_size || '', onChange: e => setEditingProfile({ ...editingProfile, parameters: { ...editingProfile.parameters, batch_size: Number(e.target.value) } }), fullWidth: true, margin: "normal", type: "number", helperText: "Batch size for processing inputs. Higher values may improve throughput but use more memory. (Default: depends on model)" }), _jsx(TextField, { label: "Top K", value: editingProfile?.parameters?.top_k || '', onChange: e => setEditingProfile({ ...editingProfile, parameters: { ...editingProfile.parameters, top_k: Number(e.target.value) } }), fullWidth: true, margin: "normal", type: "number", helperText: "Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)" }), _jsx(TextField, { label: "Top P", value: editingProfile?.parameters?.top_p || '', onChange: e => setEditingProfile({ ...editingProfile, parameters: { ...editingProfile.parameters, top_p: Number(e.target.value) } }), fullWidth: true, margin: "normal", type: "number", helperText: "Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)" }), _jsx(TextField, { label: "Minimum Probability", value: editingProfile?.parameters?.min_p || '', onChange: e => setEditingProfile({ ...editingProfile, parameters: { ...editingProfile.parameters, min_p: Number(e.target.value) } }), fullWidth: true, margin: "normal", type: "number", helperText: "Alternative to the top_p, and aims to ensure a balance of quality and variety. The parameter p represents the minimum probability for a token to be considered, relative to the probability of the most likely token. For example, with p=0.05 and the most likely token having a probability of 0.9, logits with a value less than 0.045 are filtered out. (Default: 0.0)" }), _jsxs(Accordion, { sx: { mt: 2 }, children: [_jsx(AccordionSummary, { expandIcon: _jsx(ExpandMoreIcon, {}), children: _jsxs(Box, { sx: { display: 'flex', alignItems: 'center' }, children: [_jsx(SettingsIcon, { sx: { mr: 1 } }), _jsx(Typography, { variant: "h6", children: "Circuit Breaker Configuration (Optional)" }), editingProfile?.circuit_breaker && (_jsx(Chip, { label: "Custom Settings Active", color: "primary", size: "small", sx: { ml: 2 } }))] }) }), _jsxs(AccordionDetails, { children: [_jsx(Typography, { variant: "body2", color: "text.secondary", sx: { mb: 2 }, children: "Configure timeout protection, retry behavior, and quality monitoring overrides for this specific model profile." }), _jsx(FormControlLabel, { control: _jsx(Checkbox, { checked: !!editingProfile?.circuit_breaker, onChange: (e) => {
                                                        if (e.target.checked) {
                                                            setEditingProfile({
                                                                ...editingProfile,
                                                                circuit_breaker: {}
                                                            });
                                                        }
                                                        else {
                                                            const { circuit_breaker, ...restProfile } = editingProfile;
                                                            setEditingProfile(restProfile);
                                                        }
                                                    } }), label: "Override Global Circuit Breaker Settings" }), editingProfile?.circuit_breaker && (_jsxs(_Fragment, { children: [_jsx(Typography, { variant: "subtitle1", sx: { mt: 2, mb: 1, fontWeight: 'bold' }, children: "Timeout Settings" }), _jsx(TextField, { label: "Base Timeout (seconds)", type: "number", value: editingProfile?.circuit_breaker?.base_timeout ?? '', onChange: (e) => {
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
                                                        }, fullWidth: true, margin: "normal", inputProps: { min: 1, max: 600, step: 1 }, helperText: "Base timeout for model operations (1-600 seconds)" }), _jsx(TextField, { label: "Deep Research Timeout (seconds)", type: "number", value: editingProfile?.circuit_breaker?.deep_research_timeout ?? '', onChange: (e) => {
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
                                                        }, fullWidth: true, margin: "normal", inputProps: { min: 1, max: 1200, step: 1 }, helperText: "Extended timeout for research tasks (1-1200 seconds)" }), _jsx(Typography, { variant: "subtitle1", sx: { mt: 3, mb: 1, fontWeight: 'bold' }, children: "Retry Settings" }), _jsx(TextField, { label: "Max Retries", type: "number", value: editingProfile?.circuit_breaker?.max_retries ?? '', onChange: (e) => {
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
                                                        }, fullWidth: true, margin: "normal", inputProps: { min: 0, max: 10, step: 1 }, helperText: "Maximum number of retries before giving up (0-10)" }), _jsx(TextField, { label: "Cooldown Period (seconds)", type: "number", value: editingProfile?.circuit_breaker?.cooldown_period ?? '', onChange: (e) => {
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
                                                        }, fullWidth: true, margin: "normal", inputProps: { min: 0, max: 300, step: 1 }, helperText: "Time before allowing retry after failure (0-300 seconds)" }), _jsx(Typography, { variant: "subtitle1", sx: { mt: 3, mb: 1, fontWeight: 'bold' }, children: "Quality Monitoring" }), _jsx(FormControlLabel, { control: _jsx(Switch, { checked: editingProfile?.circuit_breaker?.enable_perplexity_guard ?? true, onChange: (e) => {
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
                                                            } }), label: "Enable Perplexity Guard", sx: { mb: 1, display: 'block' } }), _jsx(TextField, { label: "Perplexity Window", type: "number", value: editingProfile?.circuit_breaker?.perplexity_window ?? '', onChange: (e) => {
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
                                                        }, fullWidth: true, margin: "normal", inputProps: { min: 10, max: 200, step: 1 }, helperText: "Number of tokens for perplexity calculation (10-200)" }), _jsx(TextField, { label: "Perplexity Threshold", type: "number", value: editingProfile?.circuit_breaker?.perplexity_threshold ?? '', onChange: (e) => {
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
                                                        }, fullWidth: true, margin: "normal", inputProps: { min: 1, max: 50, step: 0.1 }, helperText: "Perplexity threshold for quality concerns (1-50)" }), _jsx(TextField, { label: "Average Log Probability Floor", type: "number", value: editingProfile?.circuit_breaker?.avg_logprob_floor ?? '', onChange: (e) => {
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
                                                        }, fullWidth: true, margin: "normal", inputProps: { min: -20, max: 0, step: 0.1 }, helperText: "Minimum average log probability threshold (-20 to 0)" }), _jsx(Typography, { variant: "subtitle1", sx: { mt: 3, mb: 1, fontWeight: 'bold' }, children: "Repetition Detection" }), _jsx(TextField, { label: "Repetition N-gram Size", type: "number", value: editingProfile?.circuit_breaker?.repetition_ngram ?? '', onChange: (e) => {
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
                                                        }, fullWidth: true, margin: "normal", inputProps: { min: 2, max: 20, step: 1 }, helperText: "N-gram size for repetition detection (2-20)" }), _jsx(TextField, { label: "Repetition Threshold", type: "number", value: editingProfile?.circuit_breaker?.repetition_threshold ?? '', onChange: (e) => {
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
                                                        }, fullWidth: true, margin: "normal", inputProps: { min: 2, max: 20, step: 1 }, helperText: "Number of repetitions before triggering detection (2-20)" }), _jsx(TextField, { label: "Tool Generation Repetition N-gram", type: "number", value: editingProfile?.circuit_breaker?.tool_gen_repetition_ngram ?? '', onChange: (e) => {
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
                                                        }, fullWidth: true, margin: "normal", inputProps: { min: 2, max: 20, step: 1 }, helperText: "N-gram size for tool generation repetition detection (2-20)" }), _jsx(TextField, { label: "Tool Generation Repetition Threshold", type: "number", value: editingProfile?.circuit_breaker?.tool_gen_repetition_threshold ?? '', onChange: (e) => {
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
                                                        }, fullWidth: true, margin: "normal", inputProps: { min: 2, max: 20, step: 1 }, helperText: "Repetitions before triggering tool generation detection (2-20)" }), _jsx(Typography, { variant: "subtitle1", sx: { mt: 3, mb: 1, fontWeight: 'bold' }, children: "Advanced Settings" }), _jsx(TextField, { label: "Min Tokens for Evaluation", type: "number", value: editingProfile?.circuit_breaker?.min_tokens_for_eval ?? '', onChange: (e) => {
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
                                                        }, fullWidth: true, margin: "normal", inputProps: { min: 5, max: 500, step: 1 }, helperText: "Minimum tokens before starting quality evaluation (5-500)" }), _jsx(TextField, { label: "Perplexity Log Interval (tokens)", type: "number", value: editingProfile?.circuit_breaker?.perplexity_log_interval_tokens ?? '', onChange: (e) => {
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
                                                        }, fullWidth: true, margin: "normal", inputProps: { min: 5, max: 100, step: 1 }, helperText: "Interval for logging perplexity metrics (5-100 tokens)" }), _jsx(FormControlLabel, { control: _jsx(Switch, { checked: editingProfile?.circuit_breaker?.log_repetition_events ?? true, onChange: (e) => {
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
                                                            } }), label: "Log Repetition Detection Events", sx: { mt: 1, display: 'block' } })] }))] })] }), _jsxs(Accordion, { sx: { mt: 2 }, children: [_jsx(AccordionSummary, { expandIcon: _jsx(ExpandMoreIcon, {}), children: _jsxs(Box, { sx: { display: 'flex', alignItems: 'center' }, children: [_jsx(SettingsIcon, { sx: { mr: 1 } }), _jsx(Typography, { variant: "h6", children: "Parameter Optimization (Optional)" }), editingProfile?.parameter_optimization && (_jsx(Chip, { label: "Custom Settings Active", color: "primary", size: "small", sx: { ml: 2 } }))] }) }), _jsxs(AccordionDetails, { children: [_jsx(Typography, { variant: "body2", color: "text.secondary", sx: { mb: 2 }, children: "Configure automatic parameter optimization to find the maximum viable context size, batch size, and other parameters for this model profile." }), _jsx(FormControlLabel, { control: _jsx(Checkbox, { checked: !!editingProfile?.parameter_optimization, onChange: (e) => {
                                                        if (e.target.checked) {
                                                            setEditingProfile({
                                                                ...editingProfile,
                                                                parameter_optimization: createDefaultParameterOptimizationConfig()
                                                            });
                                                        }
                                                        else {
                                                            const { parameter_optimization, ...restProfile } = editingProfile;
                                                            setEditingProfile(restProfile);
                                                        }
                                                    } }), label: "Enable Parameter Optimization for this Profile" }), editingProfile?.parameter_optimization && (_jsxs(_Fragment, { children: [_jsx(FormControlLabel, { control: _jsx(Switch, { checked: editingProfile?.parameter_optimization?.enabled ?? true, onChange: (e) => {
                                                                const currentConfig = editingProfile?.parameter_optimization;
                                                                if (currentConfig !== undefined) {
                                                                    setEditingProfile({
                                                                        ...editingProfile,
                                                                        parameter_optimization: {
                                                                            ...currentConfig,
                                                                            enabled: e.target.checked
                                                                        }
                                                                    });
                                                                }
                                                            } }), label: "Enable Optimization for this Profile", sx: { mb: 2, display: 'block' } }), _jsx(Typography, { variant: "subtitle1", sx: { mt: 2, mb: 1, fontWeight: 'bold' }, children: "Parameters to Optimize" }), editingProfile?.parameter_optimization?.parameters?.map((param, index) => (_jsxs(Paper, { sx: { p: 2, mb: 2, border: '1px solid', borderColor: 'grey.300' }, children: [_jsxs(Box, { sx: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }, children: [_jsx(Typography, { variant: "h6", children: getAllParameterDisplayInfo().find(p => p.value === param.parameter_name)?.label || param.parameter_name }), _jsx(IconButton, { onClick: () => {
                                                                            const currentConfig = editingProfile?.parameter_optimization;
                                                                            if (currentConfig) {
                                                                                const newParameters = [...currentConfig.parameters];
                                                                                newParameters.splice(index, 1);
                                                                                setEditingProfile({
                                                                                    ...editingProfile,
                                                                                    parameter_optimization: {
                                                                                        ...currentConfig,
                                                                                        parameters: newParameters
                                                                                    }
                                                                                });
                                                                            }
                                                                        }, color: "error", size: "small", children: _jsx(DeleteIcon, {}) })] }), _jsxs(Box, { sx: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2, mb: 2 }, children: [_jsx(TextField, { label: "Priority", type: "number", value: param.priority, onChange: (e) => {
                                                                            const currentConfig = editingProfile?.parameter_optimization;
                                                                            if (currentConfig) {
                                                                                const newParameters = [...currentConfig.parameters];
                                                                                newParameters[index] = {
                                                                                    ...param,
                                                                                    priority: Number(e.target.value)
                                                                                };
                                                                                setEditingProfile({
                                                                                    ...editingProfile,
                                                                                    parameter_optimization: {
                                                                                        ...currentConfig,
                                                                                        parameters: newParameters
                                                                                    }
                                                                                });
                                                                            }
                                                                        }, fullWidth: true, inputProps: { min: 1 } }), _jsxs(FormControl, { fullWidth: true, children: [_jsx(InputLabel, { children: "Tuning Strategy" }), _jsxs(Select, { value: param.tuning_strategy, onChange: (e) => {
                                                                                    const currentConfig = editingProfile?.parameter_optimization;
                                                                                    if (currentConfig) {
                                                                                        const newParameters = [...currentConfig.parameters];
                                                                                        newParameters[index] = {
                                                                                            ...param,
                                                                                            tuning_strategy: e.target.value
                                                                                        };
                                                                                        setEditingProfile({
                                                                                            ...editingProfile,
                                                                                            parameter_optimization: {
                                                                                                ...currentConfig,
                                                                                                parameters: newParameters
                                                                                            }
                                                                                        });
                                                                                    }
                                                                                }, label: "Tuning Strategy", children: [_jsx(MenuItem, { value: ParameterTuningStrategyValues.BINARY_SEARCH, children: "Binary Search" }), _jsx(MenuItem, { value: ParameterTuningStrategyValues.CONSERVATIVE_INCREMENT, children: "Conservative Increment" }), _jsx(MenuItem, { value: ParameterTuningStrategyValues.EXPONENTIAL_BACKOFF, children: "Exponential Backoff" })] })] }), _jsx(TextField, { label: "Max Search Attempts", type: "number", value: param.max_search_attempts, onChange: (e) => {
                                                                            const currentConfig = editingProfile?.parameter_optimization;
                                                                            if (currentConfig) {
                                                                                const newParameters = [...currentConfig.parameters];
                                                                                newParameters[index] = {
                                                                                    ...param,
                                                                                    max_search_attempts: Number(e.target.value)
                                                                                };
                                                                                setEditingProfile({
                                                                                    ...editingProfile,
                                                                                    parameter_optimization: {
                                                                                        ...currentConfig,
                                                                                        parameters: newParameters
                                                                                    }
                                                                                });
                                                                            }
                                                                        }, fullWidth: true, inputProps: { min: 1, max: 50 } }), _jsx(TextField, { label: "Floor Value", type: "number", value: param.floor, onChange: (e) => {
                                                                            const currentConfig = editingProfile?.parameter_optimization;
                                                                            if (currentConfig) {
                                                                                const newParameters = [...currentConfig.parameters];
                                                                                newParameters[index] = {
                                                                                    ...param,
                                                                                    floor: Number(e.target.value)
                                                                                };
                                                                                setEditingProfile({
                                                                                    ...editingProfile,
                                                                                    parameter_optimization: {
                                                                                        ...currentConfig,
                                                                                        parameters: newParameters
                                                                                    }
                                                                                });
                                                                            }
                                                                        }, fullWidth: true, inputProps: { min: 1 }, helperText: "Minimum value (must be >= 1)" })] }), _jsxs(Box, { sx: { display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 2 }, children: [_jsxs(FormControl, { fullWidth: true, children: [_jsx(InputLabel, { children: "Operator" }), _jsxs(Select, { value: param.operator, onChange: (e) => {
                                                                                    const currentConfig = editingProfile?.parameter_optimization;
                                                                                    if (currentConfig) {
                                                                                        const newParameters = [...currentConfig.parameters];
                                                                                        newParameters[index] = {
                                                                                            ...param,
                                                                                            operator: e.target.value
                                                                                        };
                                                                                        setEditingProfile({
                                                                                            ...editingProfile,
                                                                                            parameter_optimization: {
                                                                                                ...currentConfig,
                                                                                                parameters: newParameters
                                                                                            }
                                                                                        });
                                                                                    }
                                                                                }, label: "Operator", children: [_jsx(MenuItem, { value: "+", children: "Add (+)" }), _jsx(MenuItem, { value: "-", children: "Subtract (-)" }), _jsx(MenuItem, { value: "*", children: "Multiply (*)" }), _jsx(MenuItem, { value: "/", children: "Divide (/)" })] })] }), _jsx(TextField, { label: "Modifier", type: "number", value: param.modifier, onChange: (e) => {
                                                                            const currentConfig = editingProfile?.parameter_optimization;
                                                                            if (currentConfig) {
                                                                                const newParameters = [...currentConfig.parameters];
                                                                                newParameters[index] = {
                                                                                    ...param,
                                                                                    modifier: Number(e.target.value)
                                                                                };
                                                                                setEditingProfile({
                                                                                    ...editingProfile,
                                                                                    parameter_optimization: {
                                                                                        ...currentConfig,
                                                                                        parameters: newParameters
                                                                                    }
                                                                                });
                                                                            }
                                                                        }, fullWidth: true, inputProps: { min: 0.1, step: 0.1 } }), _jsx(TextField, { label: "Max Value", type: "number", value: param.max_value, onChange: (e) => {
                                                                            const currentConfig = editingProfile?.parameter_optimization;
                                                                            if (currentConfig) {
                                                                                const newParameters = [...currentConfig.parameters];
                                                                                newParameters[index] = {
                                                                                    ...param,
                                                                                    max_value: Number(e.target.value)
                                                                                };
                                                                                setEditingProfile({
                                                                                    ...editingProfile,
                                                                                    parameter_optimization: {
                                                                                        ...currentConfig,
                                                                                        parameters: newParameters
                                                                                    }
                                                                                });
                                                                            }
                                                                        }, fullWidth: true, inputProps: { min: 1 } })] })] }, index))), _jsx(Box, { sx: { display: 'flex', gap: 2, mt: 2, alignItems: 'center' }, children: _jsxs(FormControl, { sx: { minWidth: 200 }, children: [_jsx(InputLabel, { children: "Add Parameter" }), _jsxs(Select, { label: "Add Parameter", onChange: (e) => {
                                                                        const currentConfig = editingProfile?.parameter_optimization;
                                                                        if (currentConfig) {
                                                                            const paramName = e.target.value;
                                                                            // Check if parameter already exists
                                                                            const exists = currentConfig.parameters.some(p => p.parameter_name === paramName);
                                                                            if (!exists) {
                                                                                setEditingProfile({
                                                                                    ...editingProfile,
                                                                                    parameter_optimization: {
                                                                                        ...currentConfig,
                                                                                        parameters: [
                                                                                            ...currentConfig.parameters,
                                                                                            createDefaultPerformanceParameter(paramName)
                                                                                        ]
                                                                                    }
                                                                                });
                                                                            }
                                                                        }
                                                                    }, displayEmpty: true, children: [_jsx(MenuItem, { value: "", disabled: true, children: "Select parameter to add..." }), getAllParameterDisplayInfo()
                                                                            .filter(param => {
                                                                            // Only show parameters that aren't already added
                                                                            const currentConfig = editingProfile?.parameter_optimization;
                                                                            if (!currentConfig) {
                                                                                return true;
                                                                            }
                                                                            return !currentConfig.parameters.some(p => p.parameter_name === param.value);
                                                                        })
                                                                            .map((param) => (_jsxs(MenuItem, { value: param.value, children: [param.label, " - ", param.description] }, param.value)))] })] }) })] }))] })] }), _jsxs(Accordion, { sx: { mt: 2 }, children: [_jsx(AccordionSummary, { expandIcon: _jsx(ExpandMoreIcon, {}), children: _jsxs(Box, { sx: { display: 'flex', alignItems: 'center' }, children: [_jsx(MemoryIcon, { sx: { mr: 1 } }), _jsx(Typography, { variant: "h6", children: "GPU Configuration (Optional)" })] }) }), _jsxs(AccordionDetails, { children: [_jsx(Alert, { severity: "info", sx: { mb: 2 }, children: _jsxs(Box, { sx: { display: 'flex', alignItems: 'center' }, children: [_jsx(WarningIcon, { sx: { mr: 1 } }), _jsx(Typography, { variant: "body2", children: "GPU configuration only applies to local models (llama.cpp, etc.). Remote API models ignore these settings." })] }) }), _jsx(FormControlLabel, { control: _jsx(Checkbox, { checked: editingProfile?.gpu_config !== undefined, onChange: (e) => {
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
                                                        }
                                                        else {
                                                            const { gpu_config, ...restProfile } = editingProfile;
                                                            setEditingProfile(restProfile);
                                                        }
                                                    } }), label: "Override Global GPU Settings" }), editingProfile?.gpu_config && (_jsxs(_Fragment, { children: [_jsx(Typography, { variant: "subtitle1", sx: { mt: 2, mb: 1, fontWeight: 'bold' }, children: "Memory Management" }), _jsx(FormControlLabel, { control: _jsx(Switch, { checked: editingProfile?.gpu_config?.no_kv_offload ?? false, onChange: (e) => {
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
                                                            } }), label: "Force KV Cache to CPU (saves VRAM)", sx: { mb: 1, display: 'block' } }), _jsx(Typography, { variant: "subtitle1", sx: { mt: 3, mb: 1, fontWeight: 'bold' }, children: "Device Selection" }), _jsx(TextField, { label: "Main GPU Device ID", value: editingProfile?.gpu_config?.main_gpu_device_id ?? '', onChange: (e) => {
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
                                                        }, fullWidth: true, margin: "normal", helperText: "GPU device ID/name (e.g., 'NVIDIA GeForce RTX 4090', empty = auto-select)" }), _jsx(TextField, { label: "Main GPU Index", type: "number", value: editingProfile?.gpu_config?.main_gpu ?? '', onChange: (e) => {
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
                                                        }, fullWidth: true, margin: "normal", helperText: "GPU device index (-1 for auto-selection, overridden by device ID above)", inputProps: { min: -1 } }), _jsxs(FormControl, { fullWidth: true, margin: "normal", children: [_jsx(InputLabel, { children: "Model Split Mode" }), _jsxs(Select, { value: editingProfile?.gpu_config?.split_mode || 'none', onChange: (e) => {
                                                                    const gpuConfig = editingProfile?.gpu_config;
                                                                    if (gpuConfig) {
                                                                        setEditingProfile({
                                                                            ...editingProfile,
                                                                            gpu_config: {
                                                                                ...gpuConfig,
                                                                                split_mode: e.target.value
                                                                            }
                                                                        });
                                                                    }
                                                                }, label: "Model Split Mode", children: [_jsx(MenuItem, { value: "none", children: "None - Single device" }), _jsx(MenuItem, { value: "layer", children: "Layer - Split by layers" }), _jsx(MenuItem, { value: "row", children: "Row - Split by tensor rows" })] })] }), _jsxs(Typography, { variant: "subtitle1", sx: { mt: 3, mb: 1, fontWeight: 'bold' }, children: ["Tensor Split Configuration", _jsx(Button, { onClick: () => {
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
                                                                }, startIcon: _jsx(AddIcon, {}), size: "small", sx: { ml: 2 }, children: "Add Device" })] }), _jsx(Typography, { variant: "body2", color: "text.secondary", sx: { mb: 2 }, children: "Distribute model computation across multiple GPUs. Values must sum to 1.0." }), editingProfile?.gpu_config?.tensor_split &&
                                                        editingProfile.gpu_config.tensor_split.length > 0 &&
                                                        (_jsxs(_Fragment, { children: [_jsx(Box, { sx: { mb: 2 }, children: (() => {
                                                                        const tensorSplit = editingProfile?.gpu_config?.tensor_split || [];
                                                                        const sum = tensorSplit.reduce((acc, val) => acc + val, 0);
                                                                        const isValid = Math.abs(sum - 1.0) < 0.01;
                                                                        return (_jsxs(Typography, { variant: "body2", color: isValid ? 'success.main' : 'error.main', children: ["Current sum: ", sum.toFixed(3), " ", isValid ? '✓' : '(must equal 1.0)'] }));
                                                                    })() }), editingProfile.gpu_config.tensor_split.map((split, index) => (_jsxs(Box, { sx: { display: 'flex', alignItems: 'center', mb: 2 }, children: [_jsxs(Typography, { sx: { minWidth: 80 }, children: ["Device ", index, ":"] }), _jsx(Slider, { value: split, onChange: (_, value) => {
                                                                                const gpuConfig = editingProfile?.gpu_config;
                                                                                if (gpuConfig && gpuConfig.tensor_split) {
                                                                                    const newTensorSplit = [...gpuConfig.tensor_split];
                                                                                    newTensorSplit[index] = value;
                                                                                    setEditingProfile({
                                                                                        ...editingProfile,
                                                                                        gpu_config: {
                                                                                            ...gpuConfig,
                                                                                            tensor_split: newTensorSplit
                                                                                        }
                                                                                    });
                                                                                }
                                                                            }, min: 0, max: 1, step: 0.01, sx: { mx: 2, flex: 1 }, valueLabelDisplay: "auto", valueLabelFormat: (value) => value.toFixed(2) }), _jsx(Typography, { sx: { minWidth: 60, textAlign: 'center' }, children: split.toFixed(2) }), _jsx(IconButton, { onClick: () => {
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
                                                                            }, size: "small", color: "error", children: _jsx(DeleteIcon, {}) })] }, index)))] })), (!editingProfile?.gpu_config?.tensor_split ||
                                                        editingProfile.gpu_config.tensor_split.length === 0) &&
                                                        (_jsx(Typography, { variant: "body2", color: "text.secondary", children: "No tensor split configured. Model will run on a single device." }))] }))] })] }), "          ", editingProfile?.circuit_breaker && (_jsx(Button, { variant: "outlined", color: "secondary", sx: { mt: 2 }, onClick: () => setEditingProfile({ ...editingProfile, circuit_breaker: undefined }), children: "Clear Circuit Breaker Config (Use Global Settings)" }))] }), _jsxs(DialogActions, { children: [_jsx(Button, { onClick: () => setDialogOpen(false), children: "Cancel" }), _jsx(Button, { onClick: () => handleSaveProfile(), variant: "contained", children: "Save" }), _jsx(Button, { onClick: () => handleSaveProfile(true), variant: "contained", children: "Save As" })] })] })] }));
};
export default ModelProfilesPage;
