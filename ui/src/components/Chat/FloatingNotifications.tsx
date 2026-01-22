import React, { useState, useEffect } from 'react';
import { Box, Alert, Fade, IconButton } from '@mui/material';
import { Close as CloseIcon } from '@mui/icons-material';

interface FloatingNotificationProps {
  message: string;
  duration?: number;
  onClose?: () => void;
}

const FloatingNotification: React.FC<FloatingNotificationProps> = ({
  message,
  duration = 4000,
  onClose
}) => {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    if (duration > 0) {
      const timer = setTimeout(() => {
        setVisible(false);
        setTimeout(() => onClose?.(), 300); // Wait for fade out animation
      }, duration);

      return () => clearTimeout(timer);
    }
  }, [duration, onClose]);

  const handleClose = () => {
    setVisible(false);
    setTimeout(() => onClose?.(), 300);
  };

  return (
    <Fade in={visible} timeout={300}>
      <Alert
        severity="info"
        variant="filled"
        action={
          <IconButton
            size="small"
            aria-label="close"
            color="inherit"
            onClick={handleClose}
          >
            <CloseIcon fontSize="small" />
          </IconButton>
        }
        sx={{
          mb: 1,
          fontSize: '0.875rem',
          '& .MuiAlert-message': {
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            maxWidth: '300px'
          }
        }}
      >
        {message}
      </Alert>
    </Fade>
  );
};

interface FloatingNotificationsProps {
  messages: string[];
  className?: string;
}

const FloatingNotifications: React.FC<FloatingNotificationsProps> = ({
  messages,
  className
}) => {
  const [notifications, setNotifications] = useState<{ id: number; message: string }[]>([]);

  useEffect(() => {
    if (messages && messages.length > 0) {
      const newNotifications = messages.map((message, index) => ({
        id: Date.now() + index,
        message
      }));
      setNotifications(prev => [...prev, ...newNotifications]);
    }
  }, [messages]);

  const removeNotification = (id: number) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };

  if (notifications.length === 0) {
    return null;
  }

  return (
    <Box
      className={className}
      sx={{
        position: 'fixed',
        top: 20,
        right: 20,
        zIndex: 1300,
        maxWidth: 400,
        pointerEvents: 'auto'
      }}
    >
      {notifications.map(notification => (
        <FloatingNotification
          key={notification.id}
          message={notification.message}
          onClose={() => removeNotification(notification.id)}
        />
      ))}
    </Box>
  );
};

export default FloatingNotifications;