

import { GraderStringCheck } from './GraderStringCheck';

import { GraderTextSimilarity } from './GraderTextSimilarity';

import { GraderPython } from './GraderPython';

import { GraderScoreModel } from './GraderScoreModel';

import { GraderLabelModel } from './GraderLabelModel';



/**
 * A MultiGrader object combines the output of multiple graders to produce a single score.
 */
export interface GraderMulti {
  /**
   * A formula to calculate the output based on grader results.
   */
  calculate_output: string;
  graders: GraderStringCheck | GraderTextSimilarity | GraderPython | GraderScoreModel | GraderLabelModel;
  /**
   * The name of the grader.
   */
  name: string;
  /**
   * The object type, which is always `multi`.
   */
  type: 'multi';
}