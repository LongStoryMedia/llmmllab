

export type ComputerEnvironment = 'windows' | 'mac' | 'linux' | 'ubuntu' | 'browser';

/**
 * Constant values for ComputerEnvironment
 */
export const ComputerEnvironmentValues = {
  /** windows */
  WINDOWS: 'windows',
  /** mac */
  MAC: 'mac',
  /** linux */
  LINUX: 'linux',
  /** ubuntu */
  UBUNTU: 'ubuntu',
  /** browser */
  BROWSER: 'browser'
} as const;