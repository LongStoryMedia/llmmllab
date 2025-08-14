import { Model } from "../types/Model";
import { ModelProfile } from "../types/ModelProfile";
import { getHeaders, req } from "./base"

export const getModels = async (accessToken: string) =>
  await req<Model[]>({
    method: 'GET',
    headers: getHeaders(accessToken),
    path: 'models'
  })

export async function listModelProfiles(token: string): Promise<ModelProfile[]> {
  return req<ModelProfile[]>({
    method: 'GET',
    path: 'models/profiles',
    headers: getHeaders(token)
  });
}

export async function getModelProfile(token: string, id: string): Promise<ModelProfile> {
  return req<ModelProfile>({
    method: 'GET',
    path: `models/profiles/${id}`,
    headers: getHeaders(token)
  });
}

export async function createModelProfile(token: string, profile: Partial<ModelProfile>): Promise<ModelProfile> {
  return req<ModelProfile>({
    method: 'POST',
    path: 'models/profiles',
    headers: getHeaders(token),
    body: JSON.stringify(profile)
  });
}

export async function updateModelProfile(token: string, id: string, profile: Partial<ModelProfile>): Promise<ModelProfile> {
  return req<ModelProfile>({
    method: 'PUT',
    path: `models/profiles/${id}`,
    headers: getHeaders(token),
    body: JSON.stringify(profile)
  });
}

export async function deleteModelProfile(token: string, id: string): Promise<void> {
  return req<void>({
    method: 'DELETE',
    path: `models/profiles/${id}`,
    headers: getHeaders(token)
  });
}