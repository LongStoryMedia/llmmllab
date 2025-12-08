import { jsx as _jsx } from "react/jsx-runtime";
import { createRoot } from 'react-dom/client';
import Wrapper from './Wrapper';
import { StrictMode } from 'react';
import { AuthProvider } from './auth';
import { BrowserRouter as Router } from 'react-router-dom';
/* @ts-expect-error ts() */
import '@fontsource/inter';
createRoot(document.getElementById('root')).render(_jsx(StrictMode, { children: _jsx(Router, { children: _jsx(AuthProvider, { children: _jsx(Wrapper, {}) }) }) }));
