import React, { useState, useEffect } from 'react';
import { Box } from '@mui/material';
import { useBackgroundContext } from '../context/BackgroundContext';
import { ImageGenerateRequest } from '../types/ImageGenerationRequest';
import { editImage, generateImage } from '../api/image';
import { useAuth } from '../auth';
import { getToken } from '../api';
import ImageGallery from '../components/Image/ImageGallery';
import ImageForm from '../components/Image/ImageForm';

const DEFAULT_SETTINGS = {
  width: 1024,
  height: 1024,
  inference_steps: 20,
  guidance_scale: 7.0,
  model: 'stable-diffusion-3'
};

const ImagePage: React.FC = () => {
  const { images, deleteImage } = useBackgroundContext();
  const { user } = useAuth();
  const [accessToken, setAccessToken] = useState<string>('');

  // Form state
  const [prompt, setPrompt] = useState<string>('');
  const [negativePrompt, setNegativePrompt] = useState<string>('');
  const [width, setWidth] = useState<number>(DEFAULT_SETTINGS.width);
  const [height, setHeight] = useState<number>(DEFAULT_SETTINGS.height);
  const [inferenceSteps, setInferenceSteps] = useState<number>(DEFAULT_SETTINGS.inference_steps);
  const [guidanceScale, setGuidanceScale] = useState<number>(DEFAULT_SETTINGS.guidance_scale);
  const [selectedModel, setSelectedModel] = useState<string>(DEFAULT_SETTINGS.model);
  
  // Image selection state
  const [selectedImage, setSelectedImage] = useState<number | null>(null);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);

  useEffect(() => {
    // Get access token for API calls
    if (user) {
      const token = getToken(user);
      setAccessToken(token);
    }
  }, [user]);

  // Get the selected image data
  const selectedImageData = selectedImage !== null
    ? images.find(img => img.id === selectedImage) || null
    : null;

  const handleGenerateImage = async () => {
    if (!prompt) {
      return;
    }

    setIsGenerating(true);

    const request: ImageGenerateRequest = {
      prompt,
      negative_prompt: negativePrompt || undefined,
      width,
      height,
      inference_steps: inferenceSteps,
      guidance_scale: guidanceScale,
      model: selectedModel
    };

    try {
      await generateImage(accessToken, request);
      // The image will be added to the images array via WebSocket notification
      setPrompt('');
      setNegativePrompt('');
    } catch (error) {
      console.error('Error generating image:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleEditImage = async () => {
    if (!prompt || !selectedImage) {
      return;
    }

    setIsGenerating(true);

    const request: ImageGenerateRequest = {
      prompt,
      negative_prompt: negativePrompt || undefined,
      width,
      height,
      inference_steps: inferenceSteps,
      guidance_scale: guidanceScale,
      model: selectedModel,
      image_id: selectedImage // Pass the ID of the image to edit
      // We would add additional parameters here for image editing if the API supports it
    };

    try {
      await editImage(accessToken, request);
      setSelectedImage(null);
      setPrompt('');
      setNegativePrompt('');
    } catch (error) {
      console.error('Error editing image:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDownload = (url?: string, name?: string) => {
    if (!url) {
      return;
    }
    
    const link = document.createElement('a');
    link.href = url;
    link.download = name ? `${name.toLowerCase()}.png` : 'image.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleSelectImage = (id: number | null) => {
    setSelectedImage(id);
    
    if (id !== null) {
      const image = images.find(img => img.id === id);
      if (image) {
        // When selecting an image, update the form with its dimensions
        setWidth(image.width || DEFAULT_SETTINGS.width);
        setHeight(image.height || DEFAULT_SETTINGS.height);
      }
    }
  };

  return (
    <Box sx={{ p: 3, display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 2 }}>
      {/* Left Panel - Image Gallery */}
      <Box sx={{ width: { xs: '100%', md: '40%', lg: '30%' } }}>
        <ImageGallery
          images={images}
          selectedImage={selectedImage}
          onSelectImage={handleSelectImage}
          onDeleteImage={deleteImage}
          onDownloadImage={handleDownload}
        />
      </Box>
      
      {/* Right Panel - Image Form */}
      <Box sx={{ width: { xs: '100%', md: '60%', lg: '70%' } }}>
        <ImageForm
          selectedImage={selectedImage}
          selectedImageData={selectedImageData}
          prompt={prompt}
          setPrompt={setPrompt}
          negativePrompt={negativePrompt}
          setNegativePrompt={setNegativePrompt}
          width={width}
          setWidth={setWidth}
          height={height}
          setHeight={setHeight}
          inferenceSteps={inferenceSteps}
          setInferenceSteps={setInferenceSteps}
          guidanceScale={guidanceScale}
          setGuidanceScale={setGuidanceScale}
          selectedModel={selectedModel}
          setSelectedModel={setSelectedModel}
          isGenerating={isGenerating}
          onGenerateImage={handleGenerateImage}
          onEditImage={handleEditImage}
          onCancelEdit={() => setSelectedImage(null)}
        />
      </Box>
    </Box>
  );
};

export default ImagePage;
