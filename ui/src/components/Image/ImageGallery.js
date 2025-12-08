import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Box, Card, CardMedia, Grid, Typography, Paper, Divider, IconButton, useTheme } from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import DeleteIcon from '@mui/icons-material/Delete';
import config from '../../config';
const ImageGallery = ({ images, selectedImage, onSelectImage, onDeleteImage, onDownloadImage }) => {
    const theme = useTheme();
    const ensureFullUrl = (url) => {
        return url.startsWith('http') ? url : `${config.server.baseUrl}${url}`;
    };
    const getImgViewUrl = (image) => {
        return ensureFullUrl(`${image.view_url || image.download_url}`);
    };
    const getImgDownloadUrl = (image) => {
        return ensureFullUrl(image.download_url ?? '');
    };
    return (_jsxs(Paper, { elevation: 3, sx: {
            p: 2,
            height: '100%',
            display: 'flex',
            flexDirection: 'column'
        }, children: [_jsx(Typography, { variant: "h6", gutterBottom: true, children: "Image Gallery" }), _jsx(Divider, { sx: { mb: 2 } }), _jsx(Box, { sx: { flexGrow: 1, overflow: 'auto', maxHeight: 'calc(100vh - 200px)' }, children: images.length === 0 ? (_jsx(Typography, { color: "text.secondary", align: "center", sx: { py: 4 }, children: "No images have been generated yet." })) : (_jsx(Grid, { container: true, spacing: 2, children: images.map((image) => {
                        return (_jsx(Grid, { sx: { width: { xs: '100%', sm: '50%', lg: '33.33%' }, padding: 1 }, children: _jsxs(Card, { sx: {
                                    cursor: 'pointer',
                                    border: selectedImage === image.id
                                        ? `2px solid ${theme.palette.primary.main}`
                                        : 'none'
                                }, onClick: () => onSelectImage(image.id ?? -1), children: [_jsx(CardMedia, { component: "img", height: "140", image: getImgViewUrl(image), alt: `Generated image: ${image.filename}`, sx: { objectFit: 'cover' } }), _jsxs(Box, { sx: {
                                            display: 'flex',
                                            justifyContent: 'space-between',
                                            p: 1
                                        }, children: [_jsx(Typography, { variant: "body2", noWrap: true, children: new Date(image.created_at).toLocaleString() }), _jsxs(Box, { children: [_jsx(IconButton, { size: "small", onClick: (e) => {
                                                            e.stopPropagation();
                                                            onDownloadImage(getImgDownloadUrl(image), image.created_at.toISOString());
                                                        }, children: _jsx(DownloadIcon, { fontSize: "small" }) }), _jsx(IconButton, { size: "small", color: "error", onClick: (e) => {
                                                            e.stopPropagation();
                                                            onDeleteImage(image.id ?? -1);
                                                            if (selectedImage === image.id) {
                                                                onSelectImage(null);
                                                            }
                                                        }, children: _jsx(DeleteIcon, { fontSize: "small" }) })] })] })] }) }, image.id));
                    }) })) })] }));
};
export default ImageGallery;
