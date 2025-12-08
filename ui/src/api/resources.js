import { getHeaders, req } from './base';
/**
 * Get device mappings with user-friendly names
 */
export async function getDeviceMappings(token) {
    return req({
        method: 'GET',
        path: 'resources/devices',
        headers: getHeaders(token)
    });
}
/**
 * Clear memory cache with various levels of aggression
 */
export async function clearMemory(token, request = {}) {
    return req({
        method: 'POST',
        path: 'resources/clear',
        headers: getHeaders(token),
        body: JSON.stringify(request)
    });
}
/**
 * Nuclear memory clear - affects all processes (admin only)
 */
export async function nuclearClearMemory(token, deviceIdx, killProcesses = true) {
    const params = new URLSearchParams();
    if (deviceIdx !== undefined) {
        params.append('device_idx', deviceIdx.toString());
    }
    params.append('kill_processes', killProcesses.toString());
    return req({
        method: 'POST',
        path: `resources/clear/nuclear?${params.toString()}`,
        headers: getHeaders(token)
    });
}
/**
 * Get GPU health and status information
 */
export async function getGpuHealth(token) {
    return req({
        method: 'GET',
        path: 'resources/health',
        headers: getHeaders(token)
    });
}
