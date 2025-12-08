import { jsx as _jsx } from "react/jsx-runtime";
import { createContext, useContext } from 'react';
import { useConfig } from '../hooks/useConfig';
const ConfigContext = createContext(null);
export const ConfigProvider = ({ children }) => {
    // Use the hook to manage configuration state
    const configState = useConfig();
    return (_jsx(ConfigContext.Provider, { value: configState, children: children }));
};
export const useConfigContext = () => {
    const context = useContext(ConfigContext);
    if (!context) {
        throw new Error('useConfigContext must be used within a ConfigProvider');
    }
    return context;
};
