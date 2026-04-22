import { Model } from "../types/Model";
import { getHeaders, req } from "./base"

export const getModels = async (accessToken: string) =>
  await req<Model[]>({
    method: 'GET',
    headers: getHeaders(accessToken),
    path: 'models'
  })
