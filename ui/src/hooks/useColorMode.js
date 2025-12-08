import React from 'react';
const isPaletteMode = (mode) => mode === 'light' || mode === 'dark';
export default function useColorMode() {
    const key = 'color-mode';
    const getInitialColorMode = () => {
        const localColorMode = localStorage.getItem(key);
        if (isPaletteMode(String(localColorMode))) {
            return String(localColorMode);
        }
        return window.matchMedia('(prefers-color-scheme: dark)').matches
            ? 'dark'
            : 'light';
    };
    const [colorMode, setColorMode] = React.useState(getInitialColorMode());
    const setMode = (mode) => {
        setColorMode(mode);
        localStorage.setItem(key, mode);
    };
    return [colorMode, setMode];
}
