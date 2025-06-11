import { type PaletteMode } from '@mui/material';
import React from 'react';

const isPaletteMode = (mode?: string): mode is PaletteMode => mode === 'light' || mode === 'dark';

type UseColorModeType = [PaletteMode, (mode: PaletteMode) => void];

export default function useColorMode(): UseColorModeType {
  const key = 'color-mode';

  const getInitialColorMode = (): PaletteMode => {
    const localColorMode = localStorage.getItem(key);

    if (isPaletteMode(String(localColorMode))) {
      return String(localColorMode) as PaletteMode;
    }

    return window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark'
      : 'light';
  };

  const [colorMode, setColorMode] = React.useState<PaletteMode>(getInitialColorMode());

  const setMode = (mode: PaletteMode) => {
    setColorMode(mode);
    localStorage.setItem(key, mode);
  };

  return [colorMode, setMode];
}
