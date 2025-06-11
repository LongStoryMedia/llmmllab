// theme.ts
import { createTheme, ThemeOptions } from '@mui/material/styles';

const commonOptions: ThemeOptions = {
  palette: {
    primary: {
      main: '#4caf50' // green
    },
    secondary: {
      main: '#81c784' // lighter green
    },
    contrastThreshold: 4.5
  },
  typography: {
    fontFamily: 'Roboto, sans-serif',
    button: {
      textTransform: 'none'
    },
    h1: {
      fontWeight: 'bold',
      fontSize: '2.6rem'
    },
    h2: {
      fontWeight: 'bold',
      fontSize: '2rem'
    },
    h3: {
      fontWeight: 'bold',
      fontSize: '1.875rem'
    },
    h4: {
      fontWeight: 'normal',
      fontSize: '1.5rem'
    },
    h5: {
      fontWeight: 'normal',
      fontSize: '1.25rem'
    },
    h6: {
      fontWeight: 'normal',
      fontSize: '1rem'
    }
  }
};

export const lightTheme = createTheme({
  ...commonOptions,
  palette: {
    ...commonOptions.palette,
    mode: 'light',
    primary: {
      main: '#00695c'
    },
    secondary: {
      main: '#5e29d9'
    },
    background: {
      default: '#f5f5f9', // Default background color
      paper: '#ffffff' // Paper color for cards, dialogs, etc.
    },
    text: {
      primary: '#1b1b1b'
    }
  }
});

export const darkTheme = createTheme({
  ...commonOptions,
  palette: {
    ...commonOptions.palette,
    mode: 'dark',
    primary: {
      main: '#00695c',
      contrastText: '#ffffff'
    },
    secondary: {
      main: '#9b7027',
      contrastText: '#0288d1'
    },
    background: {
      default: '#303030', // Dark background color
      paper: '#282828' // Paper color for cards, dialogs, etc. in dark mode
    },
    text: {
      primary: '#ffffff'
    }
  }
});
