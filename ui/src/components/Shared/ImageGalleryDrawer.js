import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useState } from 'react';
import { Drawer, Box, Typography, IconButton, Grid, Card, CardMedia, CardContent, CardActions, Button, Divider, useTheme } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import DownloadIcon from '@mui/icons-material/Download';
import DeleteIcon from '@mui/icons-material/Delete';
import { deleteImage } from '../../api/image';
import { useAuth } from '@/auth';
const ImageGalleryDrawer = ({ open, onClose, images }) => {
    const theme = useTheme();
    const [selectedImage, setSelectedImage] = useState(null);
    const { user } = useAuth();
    const handleDownload = (url, name) => {
        const link = document.createElement('a');
        link.href = url;
        link.download = name ? `${name.toLowerCase()}.png` : 'image.png';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };
    // Handler for image deletion from the gallery
    const handleRemove = (id) => {
        if (!user) {
            console.error('User not authenticated');
            return;
        }
        deleteImage(user.access_token, id);
    };
    // Handler for selecting/deselecting an image for preview
    const toggleImageSelection = (id) => {
        setSelectedImage(currentId => currentId === id ? null : id);
    };
    // Get the selected image data
    const selectedImageData = selectedImage
        ? images.find(img => img.id === selectedImage)
        : null;
    return (_jsxs(Drawer, { anchor: "right", open: open, onClose: onClose, sx: {
            '& .MuiDrawer-paper': {
                width: '100%',
                maxWidth: 600,
                p: 2
            }
        }, children: [_jsxs(Box, { sx: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }, children: [_jsx(Typography, { variant: "h6", children: "Generated Images" }), _jsx(IconButton, { onClick: onClose, children: _jsx(CloseIcon, {}) })] }), _jsx(Divider, { sx: { mb: 2 } }), images.length === 0 ? (_jsx(Typography, { color: "text.secondary", sx: { py: 4, textAlign: 'center' }, children: "No images have been generated yet." })) : (_jsxs(_Fragment, { children: [selectedImageData && (_jsxs(Box, { sx: { mb: 3 }, children: [_jsx(Typography, { variant: "subtitle1", gutterBottom: true, children: "Preview" }), _jsxs(Card, { children: [_jsx(CardMedia, { component: "img", sx: {
                                            maxHeight: '500px',
                                            objectFit: 'contain',
                                            backgroundColor: 'black'
                                        }, image: selectedImageData.view_url || selectedImageData.download_url, alt: "Selected image preview" }), _jsxs(CardActions, { children: [_jsx(Button, { startIcon: _jsx(DownloadIcon, {}), onClick: () => handleDownload(selectedImageData.download_url || '', selectedImageData.created_at.toISOString()), children: "Download" }), _jsx(Button, { startIcon: _jsx(DeleteIcon, {}), color: "error", onClick: () => {
                                                    handleRemove(selectedImage ?? -1);
                                                    setSelectedImage(null);
                                                }, children: "Remove" })] })] })] })), _jsx(Typography, { variant: "subtitle1", gutterBottom: true, children: "Image Gallery" }), _jsx(Grid, { container: true, spacing: 2, children: images.map((image) => (_jsx(Grid, { sx: { xs: 12, sm: 6, md: 4 }, children: _jsxs(Card, { sx: {
                                    cursor: 'pointer',
                                    border: selectedImage === image.id
                                        ? `2px solid ${theme.palette.primary.main}`
                                        : 'none'
                                }, onClick: () => toggleImageSelection(image.id ?? -1), children: [_jsx(CardMedia, { component: "img", height: "140", image: image.view_url || image.download_url, alt: `Generated image: ${image.filename}`, sx: { objectFit: 'cover' } }), _jsxs(CardContent, { sx: { py: 1 }, children: [_jsx(Typography, { variant: "body2", noWrap: true, title: image.filename, children: image.filename }), _jsx(Typography, { variant: "caption", color: "text.secondary", children: new Date(image.created_at).toLocaleString() })] }), _jsxs(CardActions, { children: [_jsx(Button, { size: "small", startIcon: _jsx(DownloadIcon, {}), onClick: (e) => {
                                                    e.stopPropagation(); // Prevent image selection
                                                    handleDownload(image.download_url ?? '', image.created_at.toISOString());
                                                }, children: "Download" }), _jsx(Button, { size: "small", color: "error", startIcon: _jsx(DeleteIcon, {}), onClick: (e) => {
                                                    e.stopPropagation(); // Prevent image selection
                                                    handleRemove(image.id ?? -1);
                                                    if (selectedImage === image.id) {
                                                        setSelectedImage(null);
                                                    }
                                                }, children: "Remove" })] })] }) }, image.id))) })] }))] }));
};
export default ImageGalleryDrawer;
