/**
 * Response from generating an image
 */
export interface ImageGenerateResponse {
    /**
     * Base64-encoded image
     */
    image: string;
    /**
     * URL to download the image
     */
    download: string;
}
