import { useState } from 'react';
import { Button, TextField, Typography, Paper, Box, Alert, useTheme } from '@mui/material';
import { useAuth } from '../auth';
import LoadingAnimation from '../components/Shared/LoadingAnimation';

const LoginPage = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const auth = useAuth();
  const theme = useTheme();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);
    
    try {
      await auth.userManager.getUser();
      // No need to navigate here as the AuthProvider will handle this
    } catch (error) {
      console.error('Login failed:', error);
      setError('Login failed. Please check your credentials and try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Box 
      sx={{
        height: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: theme => theme.palette.background.default
      }}
    >
      <Paper 
        elevation={3} 
        sx={{ 
          p: theme.spacing(4), 
          maxWidth: 400, 
          width: '100%',
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing(2)
        }}
      >
        <Typography variant="h4" align="center" gutterBottom>
          Welcome
        </Typography>
        
        <Typography variant="body1" align="center" color="textSecondary">
          Sign in to access your chat assistant
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mt: theme.spacing(2) }}>
            {error}
          </Alert>
        )}

        <Box component="form" onSubmit={handleLogin} sx={{ mt: theme.spacing(2) }}>
          <TextField
            label="Username"
            variant="outlined"
            fullWidth
            margin="normal"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            disabled={isLoading}
            required
          />
          <TextField
            label="Password"
            type="password"
            variant="outlined"
            fullWidth
            margin="normal"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            disabled={isLoading}
            required
          />
          <Button 
            variant="contained" 
            color="primary" 
            fullWidth 
            type="submit"
            disabled={isLoading || !username || !password}
            sx={{ mt: theme.spacing(3), mb: theme.spacing(2) }}
          >
            {isLoading ? <LoadingAnimation /> : 'Sign In'}
          </Button>
        </Box>
      </Paper>
    </Box>
  );
};

export default LoginPage;