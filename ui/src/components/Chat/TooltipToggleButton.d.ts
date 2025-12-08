import React from 'react';
interface TooltipToggleButtonProps {
    value: string;
    tooltip: string;
    disabled?: boolean;
    children: React.ReactNode;
    'aria-label': string;
    color?: 'primary' | 'secondary' | 'success' | 'error' | 'info' | 'warning' | 'standard';
    selected?: boolean;
    onSelect?: (event: React.MouseEvent<HTMLElement>, value: string) => void;
}
export declare const TooltipToggleButton: React.FC<TooltipToggleButtonProps>;
export {};
