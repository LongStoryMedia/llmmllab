import { useState } from "react";
import { useConfigContext } from "../../context/ConfigContext";
import { useChat } from "../../chat";


const useChatInput = () => {
  const [input, setInput] = useState('');
  const { sendMessage, isTyping, currentConversation, isPaused } = useChat();
  const { config, updatePartialConfig } = useConfigContext();
  const alwaysRetrieve = config?.memory?.always_retrieve || false;

  // Check if there's an active conversation
  const hasConversation = !!currentConversation?.id;

  // Get toggle states from config (default to false if not loaded)
  const summarizationEnabled = config?.summarization?.enabled !== false;
  const retrievalEnabled = config?.memory?.enabled !== false;
  const webSearchEnabled = config?.web_search?.enabled || false;
  const autoPromptRefinementEnabled = config?.image_generation?.auto_prompt_refinement || false;

  // Create array of selected options for ToggleButtonGroup
  const [selectedOptions, setSelectedOptions] = useState<string[]>(() => {
    const initialOptions: string[] = [];
    if (summarizationEnabled) {
      initialOptions.push('summarization');
    }
    if (retrievalEnabled) {
      initialOptions.push('retrieval');
    }
    if (alwaysRetrieve) {
      initialOptions.push('alwaysRetrieve');
    }
    if (webSearchEnabled) {
      initialOptions.push('webSearch');
    }
    if (autoPromptRefinementEnabled && config?.image_generation?.enabled) {
      initialOptions.push('autoPromptRefinement');
    }
    return initialOptions;
  });

  // Handler for all toggle buttons in the group
  const handleToggleChange = async (event: React.MouseEvent<HTMLElement>, newOptions: string[]) => {
    event.preventDefault();
    const wasSelected = (option: string) => selectedOptions.includes(option);
    const isSelected = (option: string) => newOptions.includes(option);

    // Update options array - this is the only place we should set selectedOptions
    setSelectedOptions(newOptions);

    // Handle summarization change
    if (wasSelected('summarization') !== isSelected('summarization')) {
      await updatePartialConfig('summarization', {
        ...config!.summarization!,
        enabled: isSelected('summarization')
      });
    }

    // Handle retrieval change
    if (wasSelected('retrieval') !== isSelected('retrieval')) {
      await updatePartialConfig('memory', {
        ...config!.memory!,
        enabled: isSelected('retrieval')
      });
    }

    // Handle always retrieve change
    if (wasSelected('alwaysRetrieve') !== isSelected('alwaysRetrieve')) {
      await updatePartialConfig('memory', {
        ...config!.memory!,
        always_retrieve: isSelected('alwaysRetrieve'),
        enabled: config?.memory?.enabled ?? false
      });
    }

    // Handle web search change
    if (wasSelected('webSearch') !== isSelected('webSearch')) {
      await updatePartialConfig('web_search', {
        ...config!.web_search!,
        enabled: isSelected('webSearch')
      });
    }

    // Handle autoPromptRefinement change
    if (wasSelected('autoPromptRefinement') !== isSelected('autoPromptRefinement') && config?.image_generation) {
      await updatePartialConfig('image_generation', {
        ...config!.image_generation!,
        auto_prompt_refinement: isSelected('autoPromptRefinement')
      });
    }
  };

  const handleSend = () => {
    const trimmedInput = input.trim();
    if (trimmedInput && !isTyping && hasConversation) {
      // Add a flag to the message metadata if image generation is requested
      sendMessage({
        content: trimmedInput,
        conversation_id: currentConversation.id!,
        metadata: {
          generate_image: selectedOptions.includes('generateImage') ? true : false,
          is_continuation: isPaused ? true : false, // indicate if this is a continuation of a paused conversation
          type: isPaused ? 'resume' : 'chat'
        }
      });
      setInput('');

      // Reset the image toggle and autoPromptRefinement after sending if needed
      if (selectedOptions.includes('generateImage') || selectedOptions.includes('autoPromptRefinement')) {
        setSelectedOptions(prev => prev.filter(option => 
          option !== 'generateImage' && option !== 'autoPromptRefinement'
        ));
      }
    }
  };

  return {
    handleSend,
    handleToggleChange,
    input,
    setInput,
    selectedOptions
  }
}

export default useChatInput;