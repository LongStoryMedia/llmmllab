

import { GraderStringCheck } from './GraderStringCheck';

import { GraderTextSimilarity } from './GraderTextSimilarity';

import { GraderPython } from './GraderPython';

import { GraderScoreModel } from './GraderScoreModel';

import { GraderMulti } from './GraderMulti';



export interface RunGraderRequest {
  /**
   * The grader used for the fine-tuning job.
   */
  grader: GraderStringCheck | GraderTextSimilarity | GraderPython | GraderScoreModel | GraderMulti;
  /**
   * The dataset item provided to the grader. This will be used to populate 
the `item` namespace. See [the guide](https://platform.openai.com/docs/guides/graders) for more details. 

   */
  item?: Record<string, unknown>;
  /**
   * The model sample to be evaluated. This value will be used to populate 
the `sample` namespace. See [the guide](https://platform.openai.com/docs/guides/graders) for more details.
The `output_json` variable will be populated if the model sample is a 
valid JSON string.
 

   */
  model_sample: string;
}