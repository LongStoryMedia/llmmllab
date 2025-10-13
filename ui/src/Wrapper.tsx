import { useMemo } from 'react'
import Router from './Router'
import { useAuth } from './auth'
import { ChatProvider } from './chat'
import { ThemeProvider } from '@emotion/react'
import useColorMode from './hooks/useColorMode'
import config from './config/index'
import { CssBaseline, styled, GlobalStyles } from '@mui/material'
import { ConfigProvider } from './context/ConfigContext'
import MainLayout from './components/Layout/MainLayout'
import ThemeToggle from './components/Shared/ThemeToggle'
import LoadingAnimation from './components/Shared/LoadingAnimation'
import { BackgroundContextProvider } from './context/BackgroundContext'

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
    overflow: 'hidden',
    height: '100%'
  },
  body: {
    overflow: 'hidden',
    height: '100%',
    margin: 0,
    padding: 0
  },
  '#root': {
    height: '100%',
    overflow: 'hidden'
  }
};


const Wrapper: React.FC = () => {
  const auth = useAuth();
  const [mode, setMode] = useColorMode();

  const theme = useMemo(() => {
    return mode === 'dark' ? config.theme.dark : config.theme.light;
  }, [mode]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <GlobalStyles styles={globalStyles} />
      <ConfigProvider>
        {auth.evaluating ? (
          <LoadingContainer>
            <LoadingAnimation size={1000} />
          </LoadingContainer>
        ) : (
          <BackgroundContextProvider>
            <ChatProvider>
              <MainLayout>
                <Router />
              </MainLayout>
            </ChatProvider>
          </BackgroundContextProvider>
        )}
      </ConfigProvider>
      <ThemeToggle mode={mode} setMode={setMode} />
    </ThemeProvider>
  )
}

export default Wrapper;
