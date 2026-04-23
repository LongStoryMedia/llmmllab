

import { SummarizationConfig } from './SummarizationConfig';

import { MemoryConfig } from './MemoryConfig';

import { ModelProfileConfig } from './ModelProfileConfig';

import { ImageGenerationConfig } from './ImageGenerationConfig';

import { WorkflowConfig } from './WorkflowConfig';

import { ToolConfig } from './ToolConfig';

import { EventStreamConfig } from './EventStreamConfig';



/**
 * User-specific configuration
 */
export interface UserConfig {
  /**
   * User ID
   */
  user_id: string;
  summarization?: SummarizationConfig;
  memory?: MemoryConfig;
  model_profiles?: ModelProfileConfig;
  image_generation?: ImageGenerationConfig;
  workflow: WorkflowConfig;
  tool?: ToolConfig;
  event_stream?: EventStreamConfig;
}