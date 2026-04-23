

import { Metadata } from './Metadata';

import { EvalCustomDataSourceConfig } from './EvalCustomDataSourceConfig';

import { EvalLogsDataSourceConfig } from './EvalLogsDataSourceConfig';

import { EvalStoredCompletionsDataSourceConfig } from './EvalStoredCompletionsDataSourceConfig';

import { EvalGraderLabelModel } from './EvalGraderLabelModel';

import { EvalGraderStringCheck } from './EvalGraderStringCheck';

import { EvalGraderTextSimilarity } from './EvalGraderTextSimilarity';

import { EvalGraderPython } from './EvalGraderPython';

import { EvalGraderScoreModel } from './EvalGraderScoreModel';



/**
 * An Eval object with a data source config and testing criteria.
An Eval represents a task to be done for your LLM integration.
Like:
 - Improve the quality of my chatbot
 - See how well my chatbot handles customer support
 - Check if o4-mini is better at my usecase than gpt-4o

 */
export interface Eval {
  /**
   * The Unix timestamp (in seconds) for when the eval was created.
   */
  created_at: number;
  /**
   * Configuration of data sources used in runs of the evaluation.
   */
  data_source_config: EvalCustomDataSourceConfig | EvalLogsDataSourceConfig | EvalStoredCompletionsDataSourceConfig;
  /**
   * Unique identifier for the evaluation.
   */
  id: string;
  metadata: Metadata;
  /**
   * The name of the evaluation.
   */
  name: string;
  /**
   * The object type.
   */
  object: 'eval';
  /**
   * A list of testing criteria.
   */
  testing_criteria: (EvalGraderLabelModel | EvalGraderStringCheck | EvalGraderTextSimilarity | EvalGraderPython | EvalGraderScoreModel)[];
}