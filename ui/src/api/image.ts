import { ImageGenerateRequest } from '../types/ImageGenerationRequest';
import { ImageGenerateResponse } from '../types/ImageGenerationResponse';
import { getHeaders, req } from './base';
import { ChatWebSocketClient } from './websocket';

/**
 * Generate an image using the Stable Diffusion API
 * @param accessToken Authentication token
 * @param request Image generation request parameters
 * @returns Promise that resolves with image data
 */
export const generateImage = async (accessToken: string, request: ImageGenerateRequest, socket?: ChatWebSocketClient) =>
  req<ImageGenerateResponse>({
    method: 'POST',
    headers: getHeaders(accessToken),
    path: 'api/images/generate',
    body: JSON.stringify(request),
    socket: socket
  });

// /**
//  * Generate an image and get real-time updates via WebSocket
//  * @param accessToken Authentication token
//  * @param request Image generation request parameters
//  * @param onUpdate Callback for WebSocket updates during generation
//  * @returns Promise that resolves with image data and a method to close the WebSocket
//  */
// export const generateImageWithUpdates = async (
//   accessToken: string,
//   request: ImageGenerateRequest,
//   onUpdate: (notification: ImageGenerationNotification) => void,
//   onError: (error: string) => void
// ): Promise<{ response: ImageGenerateResponse, closeSocket: () => void }> => {
//   // First establish the WebSocket connection to receive updates
//   const connection: WebSocketConnection = createImageGenerationSocket(
//     accessToken,
//     onUpdate,
//     onError
//   );

//   // Then trigger the image generation
//   const response = await generateImage(accessToken, request);

//   return {
//     response,
//     closeSocket: connection.close
//   };
// };

/**
 * Retrieve the URL for downloading a generated image
 * @param filename The filename of the generated image
 * @returns The URL for downloading the image
 */
export const getImageDownloadUrl = (filename: string): string => {
  return `api/images/download/${filename}`;
};