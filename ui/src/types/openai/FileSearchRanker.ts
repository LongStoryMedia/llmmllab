

/**
 * The ranker to use for the file search. If not specified will use the `auto` ranker.
 */
export type FileSearchRanker = 'auto' | 'default_2024_08_21';

/**
 * Constant values for FileSearchRanker
 */
export const FileSearchRankerValues = {
  /** auto */
  AUTO: 'auto',
  /** default_2024_08_21 */
  DEFAULT_2024_08_21: 'default_2024_08_21'
} as const;