/**
 * ModelTask represents the task associated with a machine learning model
 */
export type ModelTask = 'TextToText' | 'TextToImage' | 'ImageToText' | 'ImageToImage' | 'TextToAudio' | 'AudioToText' | 'TextToVideo' | 'VideoToText' | 'TextToSpeech' | 'SpeechToText' | 'TextToEmbeddings' | 'VisionTextToText' | 'ImageTextToImage' | 'TextToRanking';
/**
 * Constant values for ModelTask
 */
export declare const ModelTaskValues: {
    /** TextToText */
    readonly TEXTTOTEXT: "TextToText";
    /** TextToImage */
    readonly TEXTTOIMAGE: "TextToImage";
    /** ImageToText */
    readonly IMAGETOTEXT: "ImageToText";
    /** ImageToImage */
    readonly IMAGETOIMAGE: "ImageToImage";
    /** TextToAudio */
    readonly TEXTTOAUDIO: "TextToAudio";
    /** AudioToText */
    readonly AUDIOTOTEXT: "AudioToText";
    /** TextToVideo */
    readonly TEXTTOVIDEO: "TextToVideo";
    /** VideoToText */
    readonly VIDEOTOTEXT: "VideoToText";
    /** TextToSpeech */
    readonly TEXTTOSPEECH: "TextToSpeech";
    /** SpeechToText */
    readonly SPEECHTOTEXT: "SpeechToText";
    /** TextToEmbeddings */
    readonly TEXTTOEMBEDDINGS: "TextToEmbeddings";
    /** VisionTextToText */
    readonly VISIONTEXTTOTEXT: "VisionTextToText";
    /** ImageTextToImage */
    readonly IMAGETEXTTOIMAGE: "ImageTextToImage";
    /** TextToRanking */
    readonly TEXTTORANKING: "TextToRanking";
};
