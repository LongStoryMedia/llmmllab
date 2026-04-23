

import { FineTuneReinforcementHyperparameters } from './FineTuneReinforcementHyperparameters';

import { GraderStringCheck } from './GraderStringCheck';

import { GraderTextSimilarity } from './GraderTextSimilarity';

import { GraderPython } from './GraderPython';

import { GraderScoreModel } from './GraderScoreModel';

import { GraderMulti } from './GraderMulti';



/**
 * Configuration for the reinforcement fine-tuning method.
 */
export interface FineTuneReinforcementMethod {
  /**
   * The grader used for the fine-tuning job.
   */
  grader: GraderStringCheck | GraderTextSimilarity | GraderPython | GraderScoreModel | GraderMulti;
  hyperparameters?: FineTuneReinforcementHyperparameters;
}