import { jsx as _jsx } from "react/jsx-runtime";
import { ToggleButton, Tooltip, useTheme, useMediaQuery } from '@mui/material';
// Component for wrapping each toggle button with a tooltip
export const TooltipToggleButton = ({ value, tooltip, disabled = false, color = "standard", children, 'aria-label': ariaLabel, selected, onSelect }) => {
    const theme = useTheme();
    const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
    return (_jsx(Tooltip, { title: tooltip, arrow: true, placement: "bottom", enterDelay: 500, leaveDelay: 200, children: _jsx(ToggleButton, { value: value, "aria-label": ariaLabel, disabled: disabled, color: color, selected: selected, onClick: onSelect, sx: {
                padding: isMobile ? '4px 6px' : '6px 10px',
                minWidth: isMobile ? 'auto' : '100px'
            }, children: children }) }));
};
