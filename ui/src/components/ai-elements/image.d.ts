import type { Experimental_GeneratedImage } from "ai";
export type ImageProps = Experimental_GeneratedImage & {
    className?: string;
    alt?: string;
};
export declare const Image: ({ base64, uint8Array, mediaType, ...props }: ImageProps) => import("react/jsx-runtime").JSX.Element;
