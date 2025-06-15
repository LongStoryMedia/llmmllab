import { createRoot } from 'react-dom/client'
import Wrapper from './Wrapper'
import { StrictMode } from 'react'
import { AuthProvider } from './auth'
import { BrowserRouter as Router } from 'react-router-dom'
/* @ts-expect-error ts() */
import '@fontsource/inter';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Router>
      <AuthProvider>
        <Wrapper />
      </AuthProvider>
    </Router>
  </StrictMode>
)
