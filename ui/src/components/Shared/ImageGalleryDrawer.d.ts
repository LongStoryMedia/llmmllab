import React from 'react';
import { ImageMetadata } from '../../types/ImageMetadata';
interface ImageGalleryDrawerProps {
    open: boolean;
    onClose: () => void;
    images: ImageMetadata[];
}
declare const ImageGalleryDrawer: React.FC<ImageGalleryDrawerProps>;
export default ImageGalleryDrawer;
