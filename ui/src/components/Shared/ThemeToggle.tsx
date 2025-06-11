import {FormControlLabel, Switch, Theme, styled, useTheme, type PaletteMode} from '@mui/material';

// Taken straight from https://mui.com/material-ui/react-switch/#customization
// and modified slightly
const StyledSwitch = styled(Switch)(({theme}: {theme: Theme}) => ({
  width: 45,
  height: 25,
  padding: 7,
  margin: 10,
  '& .MuiSwitch-switchBase': {
    margin: 3,
    padding: 0,
    transform: 'translateX(6px)',
    '&.Mui-checked': {
      color: theme.palette.background.paper,
      transform: 'translateX(17px)',
      '& .MuiSwitch-thumb:before': {
        backgroundImage: `url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" height="12" width="12" viewBox="0 0 20 20"><path fill="${encodeURIComponent(
          theme.palette.primary.dark
        )}" d="M4.2 2.5l-.7 1.8-1.8.7 1.8.7.7 1.8.6-1.8L6.7 5l-1.9-.7-.6-1.8zm15 8.3a6.7 6.7 0 11-6.6-6.6 5.8 5.8 0 006.6 6.6z"/></svg>')`
      }
    }
  },
  '& .MuiSwitch-thumb': {
    backgroundColor: theme.palette.background.paper,
    width: 20,
    height: 20,
    '&:before': {
      content: '\'\'',
      position: 'absolute',
      width: '100%',
      height: '100%',
      left: 0,
      top: 0,
      backgroundRepeat: 'no-repeat',
      backgroundPosition: 'center',
      backgroundImage: `url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" height="15" width="15" viewBox="0 0 20 20"><path fill="${encodeURIComponent(
        theme.palette.primary.dark
      )}" d="M9.305 1.667V3.75h1.389V1.667h-1.39zm-4.707 1.95l-.982.982L5.09 6.072l.982-.982-1.473-1.473zm10.802 0L13.927 5.09l.982.982 1.473-1.473-.982-.982zM10 5.139a4.872 4.872 0 00-4.862 4.86A4.872 4.872 0 0010 14.862 4.872 4.872 0 0014.86 10 4.872 4.872 0 0010 5.139zm0 1.389A3.462 3.462 0 0113.471 10a3.462 3.462 0 01-3.473 3.472A3.462 3.462 0 016.527 10 3.462 3.462 0 0110 6.528zM1.665 9.305v1.39h2.083v-1.39H1.666zm14.583 0v1.39h2.084v-1.39h-2.084zM5.09 13.928L3.616 15.4l.982.982 1.473-1.473-.982-.982zm9.82 0l-.982.982 1.473 1.473.982-.982-1.473-1.473zM9.305 16.25v2.083h1.389V16.25h-1.39z"/></svg>')`
    }
  },
  '& .MuiSwitch-track': {
    opacity: 1,
    backgroundColor: theme.palette.background.default,
    borderRadius: 20 / 2
  }
}));

interface ThemeToggleProps {
  mode: PaletteMode;
  setMode: ( mode: PaletteMode ) => void;
}

const ThemeToggle: React.FC<ThemeToggleProps> = ({ mode, setMode }) => {
  const theme = useTheme();

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setMode(event.target.checked ? 'dark' : 'light');
  };


  return (
    <FormControlLabel
      control={
        <StyledSwitch
          checked={mode === 'dark'}
          onChange={handleChange}
          name="themeToggle"
          color="primary"
        />
      }
      label={mode === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
      style={{ color: theme.palette.text.primary }}
    />
  );
}

export default ThemeToggle;