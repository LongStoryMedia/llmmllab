

/**
 * Constrains the tools available to the model to a pre-defined set.

 */
export interface ChatCompletionAllowedTools {
  /**
   * Constrains the tools available to the model to a pre-defined set.

`auto` allows the model to pick from among the allowed tools and generate a
message.

`required` requires the model to call one or more of the allowed tools.

   */
  mode: 'auto' | 'required';
  /**
   * A list of tool definitions that the model should be allowed to call.

For the Chat Completions API, the list of tool definitions might look like:
```json
[
  { "type": "function", "function": { "name": "get_weather" } },
  { "type": "function", "function": { "name": "get_time" } }
]
```

   */
  tools: (Record<string, unknown>)[];
}