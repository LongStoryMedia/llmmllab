import React from 'react';
interface ImageModelSelectorProps {
    onModelChange?: (modelId: string) => void;
    mode: 'TextToImage' | 'ImageToImage';
}
declare const ImageModelSelector: React.FC<ImageModelSelectorProps>;
export default ImageModelSelector;
