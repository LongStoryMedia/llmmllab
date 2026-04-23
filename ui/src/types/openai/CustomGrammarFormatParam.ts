

import { GrammarSyntax1 } from './GrammarSyntax1';



/**
 * A grammar defined by the user.
 */
export interface CustomGrammarFormatParam {
  /**
   * The grammar definition.
   */
  definition: string;
  /**
   * The syntax of the grammar definition. One of `lark` or `regex`.
   */
  syntax: GrammarSyntax1;
  /**
   * Grammar format. Always `grammar`.
   */
  type: 'grammar';
}