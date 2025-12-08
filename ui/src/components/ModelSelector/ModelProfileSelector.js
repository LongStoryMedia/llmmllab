import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { FormControl, Grid, InputLabel, MenuItem, Select } from "@mui/material";
import { useEffect, useState } from "react";
import { useConfigContext } from "../../context/ConfigContext";
import { useAuth } from "../../auth";
import { getToken, listModelProfiles } from "../../api";
const ModelProfileSelector = ({ task }) => {
    const { config, updateConfig } = useConfigContext();
    const [profiles, setProfiles] = useState([]);
    const auth = useAuth();
    const [value, setValue] = useState(config?.model_profiles?.[task.key] || '');
    // Use effect to update value when config changes
    useEffect(() => {
        if (config?.model_profiles && task.key in config.model_profiles) {
            setValue(config.model_profiles[task.key] || '');
        }
    }, [config, task.key]);
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
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);
    const handleChange = (event) => {
        const newValue = event.target.value;
        setValue(newValue);
        if (config?.model_profiles) {
            updateConfig({
                ...config,
                model_profiles: {
                    ...config.model_profiles,
                    [task.key]: newValue
                }
            });
        }
    };
    return (_jsx(Grid, { size: { xs: 12, sm: 6, md: 4 }, children: _jsxs(FormControl, { fullWidth: true, children: [_jsx(InputLabel, { children: task.label }), _jsxs(Select, { value: value, onChange: handleChange, labelId: `${task.key}-select-label`, id: `${task.key}-select`, label: task.label, children: [_jsx(MenuItem, { value: "", children: "(None)" }), profiles && profiles.map(profile => (_jsx(MenuItem, { value: profile.id, children: profile.name }, profile.id)))] })] }) }, task.key));
};
export default ModelProfileSelector;
