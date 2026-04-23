

import { Metadata } from './Metadata';

import { CreateEvalJsonlRunDataSource } from './CreateEvalJsonlRunDataSource';

import { CreateEvalCompletionsRunDataSource } from './CreateEvalCompletionsRunDataSource';

import { CreateEvalResponsesRunDataSource } from './CreateEvalResponsesRunDataSource';



export interface CreateEvalRunRequest {
  /**
   * Details about the run's data source.
   */
  data_source: CreateEvalJsonlRunDataSource | CreateEvalCompletionsRunDataSource | CreateEvalResponsesRunDataSource;
  metadata?: Metadata;
  /**
   * The name of the run.
   */
  name?: string;
}