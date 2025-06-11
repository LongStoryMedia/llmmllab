import { useMemo } from 'react'
import Router from './Router'
import { useAuth } from './auth'
import { ChatProvider } from './chat'
import { ThemeProvider } from '@emotion/react'
import useColorMode from './hooks/useColorMode'
import config from './config/index'
import { CssBaseline, Box } from '@mui/material'
import { ConfigProvider } from './context/ConfigContext'
import MainLayout from './components/Layout/MainLayout'
import ThemeToggle from './components/Shared/ThemeToggle'
import LoadingAnimation from './components/Shared/LoadingAnimation'


const Wrapper:React.FC = () => {
  const auth = useAuth();
  const [mode, setMode] = useColorMode();

  const theme = useMemo(() => {
    return mode === 'dark' ? config.theme.dark : config.theme.light;
  }, [mode]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <ConfigProvider>
        {auth.evaluating ? (
          <Box sx={{ 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center', 
            width: '100%', 
            height: '100vh' 
          }}>
            <LoadingAnimation size={1000} />
          </Box>
        ) : (
          <ChatProvider>
            <MainLayout>
              <Router />
            </MainLayout>
          </ChatProvider>
        )}
      </ConfigProvider>
      <ThemeToggle mode={mode} setMode={setMode} />
    </ThemeProvider>
  )
}

export default Wrapper;
