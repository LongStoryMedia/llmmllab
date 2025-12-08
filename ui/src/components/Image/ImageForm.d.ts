import React from 'react';
import { ImageMetadata } from '../../types/ImageMetadata';
interface ImageFormProps {
    selectedImage: number | null;
    selectedImageData: ImageMetadata | null;
    prompt: string;
    setPrompt: (prompt: string) => void;
    negativePrompt: string;
    setNegativePrompt: (prompt: string) => void;
    width: number;
    setWidth: (width: number) => void;
    height: number;
    setHeight: (height: number) => void;
    inferenceSteps: number;
    setInferenceSteps: (steps: number) => void;
    guidanceScale: number;
    setGuidanceScale: (scale: number) => void;
    selectedModel: string;
    setSelectedModel: (model: string) => void;
    isGenerating: boolean;
    onGenerateImage: () => void;
    onEditImage: () => void;
    onCancelEdit: () => void;
}
declare const ImageForm: React.FC<ImageFormProps>;
export default ImageForm;
