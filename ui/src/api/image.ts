import { ImageGenerateRequest } from '../types/ImageGenerationRequest';
import { ImageGenerateResponse } from '../types/ImageGenerationResponse';
import { ImageMetadata } from '../types/ImageMetadata';
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

/**
 * Edit an existing image using the Stable Diffusion API
 * @param accessToken Authentication token
 * @param request Image edit request parameters
 * @returns Promise that resolves with image data
 */
export const editImage = async (accessToken: string, request: ImageGenerateRequest, socket?: ChatWebSocketClient) =>
  req<ImageGenerateResponse>({
    method: 'POST',
    headers: getHeaders(accessToken),
    path: `api/images/edit`,
    body: JSON.stringify(request),
    socket: socket
  });

/**
 * Fetch all images for the current user
 * @param accessToken Authentication token
 * @param limit Optional limit for number of images to return
 * @param offset Optional offset for pagination
 * @param conversationId Optional conversation ID to filter by
 * @returns Promise that resolves with an array of image metadata
 */
export const getUserImages = async (
  accessToken: string,
  limit?: number,
  offset?: number,
  conversationId?: number
) => {
  const queryParams = new URLSearchParams();
  
  if (limit) {
    queryParams.append('limit', limit.toString());
  }
  
  if (offset) {
    queryParams.append('offset', offset.toString());
  }
  
  if (conversationId) {
    queryParams.append('conversation_id', conversationId.toString());
  }
  
  const queryString = queryParams.toString();
  const path = `api/images${queryString ? `?${queryString}` : ''}`;
  
  return req<ImageMetadata[]>({
    method: 'GET',
    headers: getHeaders(accessToken),
    path
  });
};

/**
 * Delete an image by ID
 * @param accessToken Authentication token
 * @param imageId ID of the image to delete
 * @returns Promise that resolves when the image is deleted
 */
export const deleteImage = async (accessToken: string, imageId: number) => {
  return req<void>({
    method: 'DELETE',
    headers: getHeaders(accessToken),
    path: `api/images/${imageId}`
  });
};
