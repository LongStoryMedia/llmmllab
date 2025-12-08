import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "react/jsx-runtime";
import React, { useState } from 'react';
import { Fab, Badge, Tooltip } from '@mui/material';
import ImageIcon from '@mui/icons-material/Image';
import ImageGalleryDrawer from './ImageGalleryDrawer';
import { getUserImages } from '../../api/image';
import { useAuth } from '@/auth';
const GalleryFAB = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [images, setImages] = useState([]);
    const { user } = useAuth();
    React.useEffect(() => {
        const fetchImages = async () => {
            try {
                const fetchedImages = await getUserImages(user?.access_token || '');
                setImages(fetchedImages);
            }
            catch (error) {
                console.error('Error fetching images:', error);
            }
        };
        fetchImages();
    }, []);
    const handleOpen = () => {
        setIsOpen(true);
    };
    const handleClose = () => {
        setIsOpen(false);
    };
    return (_jsxs(_Fragment, { children: [_jsx(Tooltip, { title: "View Image Gallery", placement: "left", children: _jsx(Fab, { color: "secondary", "aria-label": "Open Image Gallery", onClick: handleOpen, sx: {
                        position: 'fixed',
                        bottom: 24,
                        right: 24,
                        zIndex: (theme) => theme.zIndex.drawer - 2
                    }, children: _jsx(Badge, { badgeContent: images.length, color: "error", overlap: "circular", max: 99, children: _jsx(ImageIcon, {}) }) }) }), _jsx(ImageGalleryDrawer, { open: isOpen, onClose: handleClose, images: images })] }));
};
export default GalleryFAB;
