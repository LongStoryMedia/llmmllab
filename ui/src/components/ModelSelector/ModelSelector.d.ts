import React from 'react';
import { SelectChangeEvent } from '@mui/material';
interface ModelCardProps {
    onSelect: (event: SelectChangeEvent) => void;
    name: string;
    label?: string;
    optional?: boolean;
}
declare const ModelSelector: React.FC<ModelCardProps>;
export default ModelSelector;
