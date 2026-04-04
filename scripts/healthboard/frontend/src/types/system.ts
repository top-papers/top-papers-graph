// Response structure from API endpoints
export interface ApiResponse<T> {
  status: 'success' | 'error';
  data?: T;
  message?: string;
}

export interface ApiData {}

// RAM data structure
export interface RamData extends ApiData {
  available: number;      // in GB
  total: number;     // in GB
}

// Disk data structure
export interface DiskData extends ApiData {
  free: number;      // in GB
  total: number;     // in GB
  percent_free: number; // percentage
}

// Docker service status
export interface DockerService extends ApiData {
  Service: string;
  Name: string; // container name, project-service format.
  Command: string;
  State: string;
  Ports: string;
  Health: "healthy" | "unhealthy" | "starting" | "unknown";
}

// Docker status response
export interface DockerStatus extends ApiData {
  running: "yes" | "no" | "unknown";
  services: DockerService[];
  running_count: number;
  total_count: number;
}

// Combined system data
export interface SystemData extends ApiData {
  ram: RamData | null;
  disk: DiskData | null;
  docker: DockerStatus | null;
}

// Command line utility output
export interface CliOutput extends ApiData {
  output: string;
  error?: string;
}

// SWR hook return type
export interface UseSystemDataReturn {
  ramData: ApiResponse<RamData> | undefined;
  diskData: ApiResponse<DiskData> | undefined;
  dockerData: ApiResponse<DockerStatus> | undefined;
  loading: boolean;
  error: Error | null;
  refreshAll: () => void;
  startDocker: () => Promise<void>;
  startingDocker: boolean;
  stopDocker: () => Promise<void>;
  stoppingDocker: boolean;
  createContainers: () => Promise<void>;
  creatingContainers: boolean;
  removeContainers: () => Promise<void>;
  removingContainers: boolean;
  lastAsyncError: string | undefined;
}
