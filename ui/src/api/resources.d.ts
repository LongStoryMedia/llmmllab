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
export declare function getDeviceMappings(token: string): Promise<DeviceMappingsResponse>;
/**
 * Clear memory cache with various levels of aggression
 */
export declare function clearMemory(token: string, request?: ClearMemoryRequest): Promise<ClearMemoryResponse>;
/**
 * Nuclear memory clear - affects all processes (admin only)
 */
export declare function nuclearClearMemory(token: string, deviceIdx?: number, killProcesses?: boolean): Promise<{
    detail: string;
    [key: string]: unknown;
}>;
/**
 * Get GPU health and status information
 */
export declare function getGpuHealth(token: string): Promise<HealthResponse>;
