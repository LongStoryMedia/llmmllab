import { getHeaders, req } from './base';

export interface DeviceInfo {
  index: number;
  name: string;
  uuid: string;
  id: string;
}

export interface DeviceMappingsResponse {
  devices: Record<string, DeviceInfo>;
}

export interface ClearMemoryRequest {
  device_idx?: number;
  aggressive?: boolean;
  nuclear?: boolean;
  kill_processes?: boolean;
}

export interface ClearMemoryResponse {
  detail: string;
  memory_before: Record<string, unknown>;
  memory_after: Record<string, unknown>;
  processes_killed?: Record<number, number>;
}

export interface HealthResponse {
  gpu_count: number;
  has_gpu: boolean;
  current_device: string;
  devices: Record<string, {
    name: string;
    temperature: number;
    memory_utilization: number;
    gpu_utilization: number;
    processes: number;
    status: string;
  }>;
  total_processes: number;
}

/**
 * Get device mappings with user-friendly names
 */
export async function getDeviceMappings(token: string): Promise<DeviceMappingsResponse> {
  return req<DeviceMappingsResponse>({
    method: 'GET',
    path: 'resources/devices',
    headers: getHeaders(token)
  });
}

/**
 * Clear memory cache with various levels of aggression
 */
export async function clearMemory(token: string, request: ClearMemoryRequest = {}): Promise<ClearMemoryResponse> {
  return req<ClearMemoryResponse>({
    method: 'POST',
    path: 'resources/clear',
    headers: getHeaders(token),
    body: JSON.stringify(request)
  });
}

/**
 * Nuclear memory clear - affects all processes (admin only)
 */
export async function nuclearClearMemory(token: string, deviceIdx?: number, killProcesses = true): Promise<{ detail: string; [key: string]: unknown }> {
  const params = new URLSearchParams();
  if (deviceIdx !== undefined) {
    params.append('device_idx', deviceIdx.toString());
  }
  params.append('kill_processes', killProcesses.toString());
  
  return req<{ detail: string; [key: string]: unknown }>({
    method: 'POST',
    path: `resources/clear/nuclear?${params.toString()}`,
    headers: getHeaders(token)
  });
}

/**
 * Get GPU health and status information
 */
export async function getGpuHealth(token: string): Promise<HealthResponse> {
  return req<HealthResponse>({
    method: 'GET',
    path: 'resources/health',
    headers: getHeaders(token)
  });
}