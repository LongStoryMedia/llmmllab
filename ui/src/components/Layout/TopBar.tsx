import { AppBar, Toolbar, Typography, Button, useTheme, IconButton, Box } from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import { useAuth } from '../../auth';
// import Icon from '../Shared/Icon';
// import Title from '../Shared/Title';


const TopBar = ({ onMenuClick }: { onMenuClick?: () => void }) => {
  const { user, logout } = useAuth();
  const theme = useTheme();

  return (
    <AppBar position="sticky">
      <Toolbar>
        {onMenuClick && (
          <IconButton
            color="inherit"
            edge="start"
            onClick={onMenuClick}
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>
        )}
        <Box sx={{ flexGrow: 1, display: 'flex', justifyContent: 'left' }}>
          <img
            src="/nurturebot2.png"
            alt="Logo"
            style={{ height: 75 }}
          />
          <Typography
            variant="h1"
            component="div"
            sx={{ 
              ml: 2, 
              color: theme.palette.primary.light,
              alignSelf: 'center',
              fontWeight: 300
            }}
          >
            LLLab
          </Typography>
          {/* <Icon size={80}/>
          <Title speed={15} size={100}/> */}
        </Box>
        {user?.profile.name && (
          <Typography variant="body1" sx={{ mr: theme.spacing(2.5) }}>
            Welcome, {user.profile.name}
          </Typography>
        )}
        <Button color="inherit" onClick={logout}>
          Logout
        </Button>
      </Toolbar>
    </AppBar>
  );
};

export default TopBar;