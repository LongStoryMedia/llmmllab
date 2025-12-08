import React from 'react';
interface ModelCardProps {
    modelName: string;
    modelDescription: string;
    onSelect: (model: string) => void;
}
declare const ModelCard: React.FC<ModelCardProps>;
export default ModelCard;
