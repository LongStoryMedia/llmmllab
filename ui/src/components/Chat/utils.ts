
// Utility function to replace Unicode characters that cause LaTeX compatibility issues
export const sanitizeForLaTeX = (text: string): string => {
  if (!text) {
    return '';
  }
  const replacements: Record<string, string> = {
    '\u2013': '-',
    '\u2014': '---',
    '\u2018': "'",
    '\u2019': "'",
    '\u201C': '"',
    '\u201D': '"',
    '\u2026': '...'
  };
  return Object.entries(replacements).reduce(
    (result, [unicodeChar, replacement]) =>
      result.replace(new RegExp(unicodeChar, 'g'), replacement),
    text
  );
};

export const parseResponse = (content: string, inProgress: boolean) => {
  const startIdx = content.indexOf('<think>');
  const endIdx = content.indexOf('</think>', startIdx);
  if (startIdx === -1 || (endIdx === -1 && !inProgress)) {
    // If <think> tag is not found or inProgress is false and </think> is not found
    return { think: null, rest: content };
  }
  if (endIdx !== -1) {
    const thinkContent = content.substring(startIdx + 7, endIdx).trim();
    const beforeThink = content.substring(0, startIdx).trim();
    const afterThink = content.substring(endIdx + 8).trim();
    const restContent = [beforeThink, afterThink].filter(Boolean).join('\n\n');
    return {
      think: thinkContent,
      rest: restContent || ''
    };
  } else {
    const beforeThink = content.substring(0, startIdx).trim();
    const thinkContent = content.substring(startIdx + 7).trim();
    return {
      think: thinkContent,
      rest: beforeThink || ''
    };
  }
};