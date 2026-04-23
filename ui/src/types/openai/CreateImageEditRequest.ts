

import { PartialImages } from './PartialImages';

import { InputFidelity } from './InputFidelity';



export interface CreateImageEditRequest {
  /**
   * Allows to set transparency for the background of the generated image(s).
This parameter is only supported for the GPT image models. Must be one of
`transparent`, `opaque` or `auto` (default value). When `auto` is used, the
model will automatically determine the best background for the image.

If `transparent`, the output format needs to support transparency, so it
should be set to either `png` (default value) or `webp`.

   */
  background?: 'transparent' | 'opaque' | 'auto';
  /**
   * The image(s) to edit. Must be a supported image file or an array of images.

For the GPT image models (`gpt-image-1`, `gpt-image-1-mini`, and `gpt-image-1.5`), each image should be a `png`, `webp`, or `jpg`
file less than 50MB. You can provide up to 16 images.

For `dall-e-2`, you can only provide one image, and it should be a square
`png` file less than 4MB.

   */
  image: string | (string)[];
  input_fidelity?: InputFidelity | unknown;
  /**
   * An additional image whose fully transparent areas (e.g. where alpha is zero) indicate where `image` should be edited. If there are multiple images provided, the mask will be applied on the first image. Must be a valid PNG file, less than 4MB, and have the same dimensions as `image`.
   */
  mask?: string;
  /**
   * The model to use for image generation. Only `dall-e-2` and the GPT image models are supported. Defaults to `dall-e-2` unless a parameter specific to the GPT image models is used.
   */
  model?: string | 'gpt-image-1.5' | 'dall-e-2' | 'gpt-image-1' | 'gpt-image-1-mini';
  /**
   * The number of images to generate. Must be between 1 and 10.
   */
  n?: number;
  /**
   * The compression level (0-100%) for the generated images. This parameter
is only supported for the GPT image models with the `webp` or `jpeg` output
formats, and defaults to 100.

   */
  output_compression?: number;
  /**
   * The format in which the generated images are returned. This parameter is
only supported for the GPT image models. Must be one of `png`, `jpeg`, or `webp`.
The default value is `png`.

   */
  output_format?: 'png' | 'jpeg' | 'webp';
  partial_images?: PartialImages;
  /**
   * A text description of the desired image(s). The maximum length is 1000 characters for `dall-e-2`, and 32000 characters for the GPT image models.
   */
  prompt: string;
  /**
   * The quality of the image that will be generated. `high`, `medium` and `low` are only supported for the GPT image models. `dall-e-2` only supports `standard` quality. Defaults to `auto`.

   */
  quality?: 'standard' | 'low' | 'medium' | 'high' | 'auto';
  /**
   * The format in which the generated images are returned. Must be one of `url` or `b64_json`. URLs are only valid for 60 minutes after the image has been generated. This parameter is only supported for `dall-e-2`, as the GPT image models always return base64-encoded images.
   */
  response_format?: 'url' | 'b64_json';
  /**
   * The size of the generated images. Must be one of `1024x1024`, `1536x1024` (landscape), `1024x1536` (portrait), or `auto` (default value) for the GPT image models, and one of `256x256`, `512x512`, or `1024x1024` for `dall-e-2`.
   */
  size?: '256x256' | '512x512' | '1024x1024' | '1536x1024' | '1024x1536' | 'auto';
  /**
   * Edit the image in streaming mode. Defaults to `false`. See the
[Image generation guide](https://platform.openai.com/docs/guides/image-generation) for more information.

   */
  stream?: boolean;
  /**
   * A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](https://platform.openai.com/docs/guides/safety-best-practices#end-user-ids).

   */
  user?: string;
}