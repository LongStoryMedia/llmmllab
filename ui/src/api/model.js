import { getHeaders, req } from "./base";
export const getModels = async (accessToken) => await req({
    method: 'GET',
    headers: getHeaders(accessToken),
    path: 'models'
});
export async function listModelProfiles(token) {
    return req({
        method: 'GET',
        path: 'models/profiles',
        headers: getHeaders(token)
    });
}
export async function getModelProfile(token, id) {
    return req({
        method: 'GET',
        path: `models/profiles/${id}`,
        headers: getHeaders(token)
    });
}
export async function createModelProfile(token, profile) {
    return req({
        method: 'POST',
        path: 'models/profiles',
        headers: getHeaders(token),
        body: JSON.stringify(profile)
    });
}
export async function updateModelProfile(token, id, profile) {
    return req({
        method: 'PUT',
        path: `models/profiles/${id}`,
        headers: getHeaders(token),
        body: JSON.stringify(profile)
    });
}
export async function deleteModelProfile(token, id) {
    return req({
        method: 'DELETE',
        path: `models/profiles/${id}`,
        headers: getHeaders(token)
    });
}
