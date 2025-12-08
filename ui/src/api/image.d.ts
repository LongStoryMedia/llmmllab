import { ImageGenerateRequest } from '../types/ImageGenerationRequest';
import { ImageGenerateResponse } from '../types/ImageGenerationResponse';
import { ImageMetadata } from '../types/ImageMetadata';
/**
 * Generate an image using the Stable Diffusion API
 * @param accessToken Authentication token
 * @param request Image generation request parameters
 * @returns Promise that resolves with image data
 */
export declare const generateImage: (accessToken: string, request: ImageGenerateRequest) => Promise<ImageGenerateResponse>;
/**
 * Edit an existing image using the Stable Diffusion API
 * @param accessToken Authentication token
 * @param request Image edit request parameters
 * @returns Promise that resolves with image data
 */
export declare const editImage: (accessToken: string, request: ImageGenerateRequest) => Promise<ImageGenerateResponse>;
/**
 * Fetch all images for the current user
 * @param accessToken Authentication token
 * @param limit Optional limit for number of images to return
 * @param offset Optional offset for pagination
 * @param conversationId Optional conversation ID to filter by
 * @returns Promise that resolves with an array of image metadata
 */
export declare const getUserImages: (accessToken: string, limit?: number, offset?: number, conversationId?: number) => Promise<ImageMetadata[]>;
/**
 * Delete an image by ID
 * @param accessToken Authentication token
 * @param imageId ID of the image to delete
 * @returns Promise that resolves when the image is deleted
 */
export declare const deleteImage: (accessToken: string, imageId: number) => Promise<void>;
