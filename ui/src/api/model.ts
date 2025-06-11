import { Model } from "../types/Model";
import { ModelProfile } from "../types/ModelProfile";
import { getHeaders, req } from "./base"

export const getModels = async (accessToken: string) =>
  await req<{ models: Model[] }>({
    method: 'GET',
    headers: getHeaders(accessToken),
    path: 'api/models'
  })

export async function listModelProfiles(token: string): Promise<ModelProfile[]> {
  return req<ModelProfile[]>({
    method: 'GET',
    path: 'api/model-profiles',
    headers: getHeaders(token)
  });
}

export async function getModelProfile(token: string, id: string): Promise<ModelProfile> {
  return req<ModelProfile>({
    method: 'GET',
    path: `api/model-profiles/${id}`,
    headers: getHeaders(token)
  });
}

export async function createModelProfile(token: string, profile: Partial<ModelProfile>): Promise<ModelProfile> {
  return req<ModelProfile>({
    method: 'POST',
    path: 'api/model-profiles',
    headers: getHeaders(token),
    body: JSON.stringify(profile)
  });
}

export async function updateModelProfile(token: string, id: string, profile: Partial<ModelProfile>): Promise<ModelProfile> {
  return req<ModelProfile>({
    method: 'PUT',
    path: `api/model-profiles/${id}`,
    headers: getHeaders(token),
    body: JSON.stringify(profile)
  });
}

export async function deleteModelProfile(token: string, id: string): Promise<void> {
  return req<void>({
    method: 'DELETE',
    path: `api/model-profiles/${id}`,
    headers: getHeaders(token)
  });
}