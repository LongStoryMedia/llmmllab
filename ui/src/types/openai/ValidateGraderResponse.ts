

import { GraderStringCheck } from './GraderStringCheck';

import { GraderTextSimilarity } from './GraderTextSimilarity';

import { GraderPython } from './GraderPython';

import { GraderScoreModel } from './GraderScoreModel';

import { GraderMulti } from './GraderMulti';



export interface ValidateGraderResponse {
  /**
   * The grader used for the fine-tuning job.
   */
  grader?: GraderStringCheck | GraderTextSimilarity | GraderPython | GraderScoreModel | GraderMulti;
}