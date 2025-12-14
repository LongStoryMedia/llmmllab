import { useMemo, useEffect } from 'react'
import Router from './Router'
import { useAuth } from './auth'
import { ChatProvider } from './chat'
import { ThemeProvider } from '@emotion/react'
import useColorMode from './hooks/useColorMode'
import config from './config/index'
import { CssBaseline, styled, GlobalStyles, Theme, useTheme } from '@mui/material'
import { ConfigProvider } from './context/ConfigContext'
import MainLayout from './components/Layout/MainLayout'
import LoadingAnimation from './components/Shared/LoadingAnimation'

const isChatPage = (path: string) => {
  return path === '/' || path.startsWith('/chat/');
};

// Utility function to convert any color to HSL format
const colorToHsl = (color: string): string => {
  // Create a temporary element to get computed color
  const temp = document.createElement('div');
  temp.style.color = color;
  document.body.appendChild(temp);
  const computedColor = getComputedStyle(temp).color;
  document.body.removeChild(temp);

  // Parse RGB values
  const match = computedColor.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
  if (!match) {
    return '0 0% 50%';
  }

  const r = parseInt(match[1]) / 255;
  const g = parseInt(match[2]) / 255;
  const b = parseInt(match[3]) / 255;

  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  let h = 0, s = 0;
  const l = (max + min) / 2;

  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    switch (max) {
      case r: h = (g - b) / d + (g < b ? 6 : 0); break;
      case g: h = (b - r) / d + 2; break;
      case b: h = (r - g) / d + 4; break;
    }
    h /= 6;
  }

  return `${Math.round(h * 360)} ${Math.round(s * 100)}% ${Math.round(l * 100)}%`;
};

// Generate lighter/darker variants of a color
const adjustLightness = (hsl: string, adjustment: number): string => {
  const [h, s, l] = hsl.split(' ').map((v, i) =>
    i === 2 ? Math.max(0, Math.min(100, parseInt(v) + adjustment)) : v
  );
  return `${h} ${s} ${l}%`;
};

// Dynamic theme-to-CSS-variables conversion
const generateCssVariables = (theme: Theme) => {
  const isDark = theme.palette.mode === 'dark';

  // Core colors from MUI theme
  const background = colorToHsl(theme.palette.background.default);
  const paper = colorToHsl(theme.palette.background.paper);
  const text = colorToHsl(theme.palette.text.primary);
  const textSecondary = colorToHsl(theme.palette.text.secondary || theme.palette.text.disabled || '#666');
  const primary = colorToHsl(theme.palette.primary.main);
  const primaryContrast = colorToHsl(theme.palette.primary.contrastText || '#fff');

  return {
    // Base colors directly from MUI
    background,
    foreground: text,
    card: paper,
    cardForeground: text,
    primary,
    primaryForeground: primaryContrast,

    // Derived colors with smart defaults
    muted: isDark ? adjustLightness(background, 5) : adjustLightness(background, -3),
    mutedForeground: textSecondary,
    border: isDark ? adjustLightness(background, 10) : adjustLightness(background, -8),
    input: isDark ? adjustLightness(background, 8) : adjustLightness(background, -5),
    ring: primary,
    accent: isDark ? adjustLightness(background, 8) : adjustLightness(background, -5),
    accentForeground: text,

    // Semantic colors
    destructive: isDark ? '0 62% 30%' : '0 84% 60%',
    destructiveForeground: isDark ? '0 0% 98%' : '0 0% 98%'
  };
};

const LoadingContainer = styled('div')(({ theme }) => ({
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  width: '100%',
  height: '100vh',
  position: 'fixed',
  top: 0,
  left: 0,
  backgroundColor: theme.palette.background.default,
  zIndex: 1300
}));

const globalStyles = {
  html: {
    // overflow: 'hidden',
    height: '100%'
  },
  body: {
    // overflow: 'hidden',
    height: '100%',
    margin: 0,
    padding: 0
  },
  '#root': {
    height: '100%',
    overflow: isChatPage(window.location.pathname) ? 'hidden' : 'auto'
  }
};


const Wrapper: React.FC = () => {
  const auth = useAuth();
  const [mode] = useColorMode();
  const theme = useTheme();

  const themeMode = useMemo(() => {
    return mode === 'dark' ? config.theme.dark : config.theme.light;
  }, [mode]);

  // Set CSS variables based on MUI theme 
  useEffect(() => {
    const root = document.documentElement;
    const colors = generateCssVariables(themeMode);

    // Apply all CSS variables dynamically
    Object.entries(colors).forEach(([key, value]) => {
      const cssVarName = `--${key.replace(/([A-Z])/g, '-$1').toLowerCase()}`;
      root.style.setProperty(cssVarName, value);
    });

    // Set additional design tokens
    root.style.setProperty('--radius', theme.shape.borderRadius + 'px');

    // Update body class for dark mode
    const isDark = themeMode.palette.mode === 'dark';
    document.body.classList.toggle('dark', isDark);
  }, [themeMode, theme]); // Only depend on theme, not mode separately

  return (
    <ThemeProvider theme={themeMode}>
      <CssBaseline />
      <GlobalStyles styles={globalStyles} />
      <ConfigProvider>
        {auth.evaluating ? (
          <LoadingContainer>
            <LoadingAnimation size={1000} />
          </LoadingContainer>
        ) : (
          <ChatProvider>
            <MainLayout>
              <Router />
            </MainLayout>
          </ChatProvider>
        )}
      </ConfigProvider>
    </ThemeProvider>
  )
}

export default Wrapper;
