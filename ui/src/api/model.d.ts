import { Model } from "../types/Model";
import { ModelProfile } from "../types/ModelProfile";
export declare const getModels: (accessToken: string) => Promise<Model[]>;
export declare function listModelProfiles(token: string): Promise<ModelProfile[]>;
export declare function getModelProfile(token: string, id: string): Promise<ModelProfile>;
export declare function createModelProfile(token: string, profile: Partial<ModelProfile>): Promise<ModelProfile>;
export declare function updateModelProfile(token: string, id: string, profile: Partial<ModelProfile>): Promise<ModelProfile>;
export declare function deleteModelProfile(token: string, id: string): Promise<void>;
