import React from 'react';
import { ImageMetadata } from '../../types/ImageMetadata';
interface ImageGalleryProps {
    images: ImageMetadata[];
    selectedImage: number | null;
    onSelectImage: (id: number | null) => void;
    onDeleteImage: (id: number) => void;
    onDownloadImage: (url?: string, name?: string) => void;
}
declare const ImageGallery: React.FC<ImageGalleryProps>;
export default ImageGallery;
