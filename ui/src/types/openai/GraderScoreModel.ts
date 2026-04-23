

import { EvalItem } from './EvalItem';

import { ReasoningEffort } from './ReasoningEffort';



/**
 * The sampling parameters for the model.
 */
export interface SamplingParams {
  max_completions_tokens?: number | unknown;
  reasoning_effort?: ReasoningEffort;
  seed?: number | unknown;
  temperature?: number | unknown;
  top_p?: number | unknown;
}

/**
 * A ScoreModelGrader object that uses a model to assign a score to the input.

 */
export interface GraderScoreModel {
  /**
   * The input messages evaluated by the grader. Supports text, output text, input image, and input audio content blocks, and may include template strings.

   */
  input: (EvalItem)[];
  /**
   * The model to use for the evaluation.
   */
  model: string;
  /**
   * The name of the grader.
   */
  name: string;
  /**
   * The range of the score. Defaults to `[0, 1]`.
   */
  range?: (number)[];
  /**
   * The sampling parameters for the model.
   */
  sampling_params?: SamplingParams;
  /**
   * The object type, which is always `score_model`.
   */
  type: 'score_model';
}