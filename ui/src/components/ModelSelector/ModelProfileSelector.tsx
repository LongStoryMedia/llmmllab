import { FormControl, Grid, InputLabel, MenuItem, Select, SelectChangeEvent } from "@mui/material";
import { useEffect, useState } from "react";
import { useConfigContext } from "../../context/ConfigContext";
import { useAuth } from "../../auth";
import { getToken, listModelProfiles } from "../../api";
import { ModelProfileConfig } from "../../types/ModelProfileConfig";
import { ModelProfile } from "../../types/ModelProfile";

const ModelProfileSelector = ({ task }: { task: { key: keyof ModelProfileConfig; label: string; }}) => {
  const { config, updateConfig } = useConfigContext();
  const [profiles, setProfiles] = useState<ModelProfile[]>([]);
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
      } catch (err: unknown) {
        if (err instanceof Error) {
          console.error('Error fetching model profiles:', err.message);
        }
      }
    };
    fetchProfiles();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  
  const handleChange = (event: SelectChangeEvent) => {
    const newValue = event.target.value as string;
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
  return (
    <Grid key={task.key} size={{ xs: 12, sm: 6, md: 4 }}>
      <FormControl fullWidth>
        <InputLabel>{task.label}</InputLabel>
        <Select
          value={value}
          onChange={handleChange}
          labelId={`${task.key}-select-label`}
          id={`${task.key}-select`}
          label={task.label}
        >
          <MenuItem value="">(None)</MenuItem>
          {profiles && profiles.map(profile => (
            <MenuItem key={profile.id} value={profile.id}>{profile.name}</MenuItem>
          ))}
        </Select>
      </FormControl>
    </Grid>
  )
}

export default ModelProfileSelector;