import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useState, useEffect } from 'react';
import { Box, TextField, Typography, Button, Alert, Grid, Accordion, AccordionSummary, AccordionDetails, IconButton } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { addUser, deleteUser, getAllUserInfo, updatePassword } from '../../api/usrmgr';
import { useAuth } from '../../auth';
import LoadingAnimation from '../Shared/LoadingAnimation';
import DeleteIcon from '@mui/icons-material/Delete';
const SecuritySettings = () => {
    const [allUserInfo, setAllUserInfo] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [saveStatus, setSaveStatus] = useState(null);
    const [passwords, setPasswords] = useState({ oldPassword: '', newPassword: '' });
    const [newUser, setNewUser] = useState({
        Username: '',
        Password: '',
        CN: '',
        Mail: ''
    });
    const [isSaving, setIsSaving] = useState(false);
    const { user, isAdmin, userInfo } = useAuth();
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
            }
            else {
                setSaveStatus({ success: false, message: res.message || 'Failed to update password.' });
            }
        }
        catch (err) {
            setSaveStatus({ success: false, message: `Error updating password: ${err instanceof Error ? err.message : String(err)}` });
        }
        finally {
            setIsSaving(false);
        }
    };
    const handleAddUser = async (newUser) => {
        setSaveStatus(null);
        setIsSaving(true);
        try {
            const res = await addUser(newUser);
            if (res.success) {
                setSaveStatus({ success: true, message: 'User added successfully!' });
            }
            else {
                setSaveStatus({ success: false, message: res.message || 'Failed to add user.' });
            }
        }
        catch (err) {
            setSaveStatus({ success: false, message: `Error adding user: ${err instanceof Error ? err.message : String(err)}` });
        }
        finally {
            setIsSaving(false);
        }
    };
    const handleDeleteUser = async (userId) => {
        setSaveStatus(null);
        setIsSaving(true);
        try {
            await deleteUser(userId);
            setSaveStatus({ success: true, message: 'User deleted successfully!' });
            // Refresh user list after deletion
            const updatedUsers = await getAllUserInfo();
            setAllUserInfo(updatedUsers);
        }
        catch (err) {
            setSaveStatus({ success: false, message: `Error deleting user: ${err instanceof Error ? err.message : String(err)}` });
        }
        finally {
            setIsSaving(false);
        }
    };
    if (isLoading) {
        return _jsx(LoadingAnimation, { size: 500 });
    }
    return (_jsxs(Box, { sx: { padding: 2 }, children: [_jsx(Typography, { variant: "h4", gutterBottom: true, children: "Security Settings" }), saveStatus && (_jsx(Alert, { severity: saveStatus.success ? "success" : "error", sx: { mb: 2 }, onClose: () => setSaveStatus(null), children: saveStatus.message })), _jsx(Typography, { variant: "subtitle2", gutterBottom: true, align: 'left', children: "User Info" }), _jsx(Grid, { container: true, spacing: 2, sx: { mb: 2 }, children: userInfo?.Attributes.map(attr => (_jsx(Grid, { children: _jsx(TextField, { label: attr.Name, value: Array.isArray(attr.Values) ? attr.Values.join(', ') : attr.Values, fullWidth: true, margin: "normal", disabled: true }) }, attr.Name))) }), _jsx(Typography, { variant: "subtitle2", gutterBottom: true, sx: { mt: 2 }, align: 'left', children: "Change Password" }), _jsx(TextField, { label: "Current Password", type: "password", value: passwords.oldPassword, onChange: e => setPasswords(p => ({ ...p, oldPassword: e.target.value })), fullWidth: true, margin: "normal", helperText: "Enter your current password" }), _jsx(TextField, { label: "New Password", type: "password", value: passwords.newPassword, onChange: e => setPasswords(p => ({ ...p, newPassword: e.target.value })), fullWidth: true, margin: "normal", helperText: "Enter your new password" }), _jsx(Button, { variant: "contained", color: "primary", sx: { mt: 2 }, onClick: handlePasswordChange, disabled: isSaving || !passwords.oldPassword || !passwords.newPassword, children: isSaving ? 'Updating...' : 'Update Password' }), isAdmin && (_jsxs(_Fragment, { children: [_jsx(Typography, { variant: "subtitle1", gutterBottom: true, sx: { mt: 4 }, children: "Admin Settings" }), _jsxs(Accordion, { children: [_jsx(AccordionSummary, { expandIcon: _jsx(ExpandMoreIcon, {}), children: "Add New User" }), _jsx(AccordionDetails, { children: _jsx(Typography, { variant: "body2", color: "textSecondary", children: "Fill in the details below to add a new user." }) }), _jsx(TextField, { label: "New User Email", type: "email", fullWidth: true, onChange: e => setNewUser(nu => ({ ...nu, Mail: e.target.value })), value: newUser.Mail, margin: "normal", helperText: "Enter the email of the new user to add" }), _jsx(TextField, { label: "New User Name", type: "text", fullWidth: true, onChange: e => setNewUser(nu => ({ ...nu, Username: e.target.value })), value: newUser.Username, margin: "normal", helperText: "Enter the name of the new user" }), _jsx(TextField, { label: "New User Password", type: "password", fullWidth: true, onChange: e => setNewUser(nu => ({ ...nu, Password: e.target.value })), value: newUser.Password, margin: "normal", helperText: "Enter a password for the new user" }), _jsx(TextField, { label: "Full Name", type: "text", fullWidth: true, onChange: e => setNewUser(nu => ({ ...nu, CN: e.target.value })), value: newUser.CN, margin: "normal", helperText: "Enter the full name of the new user" }), _jsx(Button, { variant: "contained", color: "primary", sx: { mt: 2 }, onClick: () => {
                                    if (newUser.Mail && newUser.Username && newUser.Password && newUser.CN) {
                                        handleAddUser(newUser);
                                        setNewUser({ Username: '', Password: '', CN: '', Mail: '' }); // Reset form
                                    }
                                    else {
                                        setSaveStatus({ success: false, message: 'Please fill in all fields.' });
                                    }
                                }, disabled: isSaving, children: isSaving ? _jsx(LoadingAnimation, { size: 50, speed: 1 }) : 'Add User' })] }), _jsx(Typography, { variant: "subtitle2", gutterBottom: true, sx: { mt: 2 }, align: 'left', children: "All Users" }), _jsx(Grid, { container: true, spacing: 2, children: allUserInfo.map((user, index) => {
                            const cnAttr = user.Attributes.find(attr => attr.Name === 'cn');
                            return (_jsx(Grid, { children: _jsxs(Accordion, { children: [_jsx(AccordionSummary, { expandIcon: _jsx(ExpandMoreIcon, {}), children: cnAttr ? cnAttr.Values : 'User' }), _jsx(AccordionDetails, { children: _jsxs(Box, { component: "ul", sx: { pl: 2, mb: 0, textAlign: 'left' }, children: [user.Attributes.map(attr => (_jsxs("li", { children: [_jsxs("strong", { children: [attr.Name, ":"] }), " ", Array.isArray(attr.Values) ? attr.Values.join(', ') : attr.Values] }, attr.Name))), _jsx(IconButton, { onClick: () => handleDeleteUser(String(user.Attributes.find(attr => attr.Name === 'uid')?.Values?.[0])), children: _jsx(DeleteIcon, {}) })] }) })] }) }, index));
                        }) })] }))] }));
};
export default SecuritySettings;
