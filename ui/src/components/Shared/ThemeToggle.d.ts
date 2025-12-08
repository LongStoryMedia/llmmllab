import { type PaletteMode } from '@mui/material';
interface ThemeToggleProps {
    mode: PaletteMode;
    setMode: (mode: PaletteMode) => void;
}
declare const ThemeToggle: React.FC<ThemeToggleProps>;
export default ThemeToggle;
