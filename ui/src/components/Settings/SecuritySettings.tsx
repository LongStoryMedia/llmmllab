import { useState, useEffect } from 'react';
import { Box, TextField, Typography, Button, Alert, Grid, Accordion, AccordionSummary, AccordionDetails, IconButton } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { addUser, deleteUser, getAllUserInfo, updatePassword } from '../../api/usrmgr';
import { NewUserReq, UserInfo } from '../../api';
import { useAuth } from '../../auth';
import LoadingAnimation from '../Shared/LoadingAnimation';
import DeleteIcon from '@mui/icons-material/Delete';
  
const SecuritySettings = () => {
  const [allUserInfo, setAllUserInfo] = useState<UserInfo[]>([]); 
  const [isLoading, setIsLoading] = useState(true);
  const [saveStatus, setSaveStatus] = useState<{success?: boolean; message: string} | null>(null);
  const [passwords, setPasswords] = useState({ oldPassword: '', newPassword: '' });
  const [newUser, setNewUser] = useState<NewUserReq>({
    Username: '',
    Password: '',
    CN: '',
    Mail: ''
  });
  const [isSaving, setIsSaving] = useState(false);
  const {user, isAdmin, userInfo} = useAuth();

  useEffect(() => {
    setIsLoading(true);
    (async () => {
      if (isAdmin) {
        const allUsers = await getAllUserInfo();
        setAllUserInfo(allUsers);
      }

      setIsLoading(false);
    })();
    
  }, [user, isAdmin]);

  const handlePasswordChange = async () => {
    setSaveStatus(null);
    setIsSaving(true);
    try {
      if (!userInfo) {
        setSaveStatus({ success: false, message: 'User info not loaded.' });
        return;
      }
      const res = await updatePassword(passwords.oldPassword, passwords.newPassword);
      if (res.success) {
        setSaveStatus({ success: true, message: 'Password updated successfully!' });
        setPasswords({ oldPassword: '', newPassword: '' });
      } else {
        setSaveStatus({ success: false, message: res.message || 'Failed to update password.' });
      }
    } catch (err) {
      setSaveStatus({ success: false, message: `Error updating password: ${err instanceof Error ? err.message : String(err)}` });
    } finally {
      setIsSaving(false);
    }
  };

  const handleAddUser = async (newUser: NewUserReq) => {
    setSaveStatus(null);
    setIsSaving(true);
    try {
      const res = await addUser(newUser);
      if (res.success) {
        setSaveStatus({ success: true, message: 'User added successfully!' });
      } else {
        setSaveStatus({ success: false, message: res.message || 'Failed to add user.' });
      }
    } catch (err) {
      setSaveStatus({ success: false, message: `Error adding user: ${err instanceof Error ? err.message : String(err)}` });
    } finally {
      setIsSaving(false);
    }
  };

  const handleDeleteUser = async (userId: string) => {
    setSaveStatus(null);
    setIsSaving(true);
    try {
      await deleteUser(userId);
      setSaveStatus({ success: true, message: 'User deleted successfully!' });
      // Refresh user list after deletion
      const updatedUsers = await getAllUserInfo();
      setAllUserInfo(updatedUsers);
    } catch (err) {
      setSaveStatus({ success: false, message: `Error deleting user: ${err instanceof Error ? err.message : String(err)}` });
    } finally {
      setIsSaving(false);
    }
  };

  if (isLoading) {
    return <LoadingAnimation size={500} />;
  }

  return (
    <Box sx={{ padding: 2 }}>
      <Typography variant="h4" gutterBottom>
        Security Settings
      </Typography>

      {saveStatus && (
        <Alert
          severity={saveStatus.success ? "success" : "error"}
          sx={{ mb: 2 }}
          onClose={() => setSaveStatus(null)}
        >
          {saveStatus.message}
        </Alert>
      )}

      <Typography variant="subtitle2" gutterBottom align='left'>
        User Info
      </Typography>

      <Grid container spacing={2} sx={{ mb: 2 }}>
        {
          userInfo?.Attributes.map(attr => (
            <Grid key={attr.Name}>
              <TextField
                label={attr.Name}
                value={Array.isArray(attr.Values) ? attr.Values.join(', ') : attr.Values}
                fullWidth
                margin="normal"
                disabled
              />
            </Grid>
          ))
        }
      </Grid> 

      <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }} align='left'>
        Change Password
      </Typography>

      <TextField
        label="Current Password"
        type="password"
        value={passwords.oldPassword}
        onChange={e => setPasswords(p => ({ ...p, oldPassword: e.target.value }))}
        fullWidth
        margin="normal"
        helperText="Enter your current password"
      />

      <TextField
        label="New Password"
        type="password"
        value={passwords.newPassword}
        onChange={e => setPasswords(p => ({ ...p, newPassword: e.target.value }))}
        fullWidth
        margin="normal"
        helperText="Enter your new password"
      />

      <Button
        variant="contained"
        color="primary"
        sx={{ mt: 2 }}
        onClick={handlePasswordChange}
        disabled={isSaving || !passwords.oldPassword || !passwords.newPassword}
      >
        {isSaving ? 'Updating...' : 'Update Password'}
      </Button>

      {isAdmin && (<>
        <Typography variant="subtitle1" gutterBottom sx={{ mt: 4 }}>
          Admin Settings
        </Typography>
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            Add New User
          </AccordionSummary>
          <AccordionDetails>
            <Typography variant="body2" color="textSecondary">
              Fill in the details below to add a new user.
            </Typography>
          </AccordionDetails>
          <TextField
            label="New User Email"
            type="email"
            fullWidth
            onChange={e => setNewUser(nu => ({ ...nu, Mail: e.target.value }))}
            value={newUser.Mail}
            margin="normal"
            helperText="Enter the email of the new user to add"
          />
          <TextField
            label="New User Name"
            type="text"
            fullWidth
            onChange={e => setNewUser(nu => ({ ...nu, Username: e.target.value }))}
            value={newUser.Username}
            margin="normal"
            helperText="Enter the name of the new user"
          />
          <TextField
            label="New User Password"
            type="password"
            fullWidth
            onChange={e => setNewUser(nu => ({ ...nu, Password: e.target.value }))}
            value={newUser.Password}
            margin="normal"
            helperText="Enter a password for the new user"
          />
          <TextField
            label="Full Name"
            type="text"
            fullWidth
            onChange={e => setNewUser(nu => ({ ...nu, CN: e.target.value }))}
            value={newUser.CN}
            margin="normal"
            helperText="Enter the full name of the new user"
          />  
          <Button
            variant="contained"
            color="primary"
            sx={{ mt: 2 }}
            onClick={() => {
              if (newUser.Mail && newUser.Username && newUser.Password && newUser.CN) {
                handleAddUser(newUser);
                setNewUser({ Username: '', Password: '', CN: '', Mail: '' }); // Reset form
              } else {
                setSaveStatus({ success: false, message: 'Please fill in all fields.' });
              }
            }} 
            disabled={isSaving}
          >
            {isSaving ? <LoadingAnimation size={50} speed={1}/> : 'Add User'}
          </Button>
        </Accordion>

        <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }} align='left'>
          All Users
        </Typography>

        <Grid container spacing={2}>
          {allUserInfo.map((user, index) => {
            const cnAttr = user.Attributes.find(attr => attr.Name === 'cn');
            return (
              <Grid key={index}>
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    {cnAttr ? cnAttr.Values : 'User'}
                  </AccordionSummary>
                  <AccordionDetails>
                    <Box component="ul" sx={{ pl: 2, mb: 0, textAlign: 'left' }}>
                      {user.Attributes.map(attr => (
                        <li key={attr.Name}>
                          <strong>{attr.Name}:</strong> {Array.isArray(attr.Values) ? attr.Values.join(', ') : attr.Values}
                        </li>
                      ))}
                      <IconButton onClick={() => handleDeleteUser(String(user.Attributes.find(attr => attr.Name === 'uid')?.Values?.[0]))}><DeleteIcon /></IconButton>
                    </Box>
                  </AccordionDetails>
                </Accordion>
              </Grid>
            );
          })}
        </Grid>
      </>)}
    </Box>
  );
};

export default SecuritySettings;