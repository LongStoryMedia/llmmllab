#!/usr/bin/env python3
"""
Test harmony channel processing logic for the new OpenAI GPT OSS format
"""

import re

def filter_harmony_channels_v2(content: str) -> str:
    """Filter harmony channels for OpenAI GPT OSS format.
    
    Format: <|start|>assistant<|channel|>[channel_name]<|message|>[content]<|end|>
    """
    if '<|start|>assistant<|channel|>' not in content:
        return content
        
    print(f"Harmony filtering v2: processing {len(content)} chars")
    
    # Extract observer text before first harmony channel
    observer_text = ""
    first_harmony_match = re.search(r'<\|start\|>assistant<\|channel\|>', content)
    if first_harmony_match:
        observer_text = content[:first_harmony_match.start()].strip()
        print(f"Extracted observer text: {repr(observer_text[:100])}")
    
    # Extract all complete harmony channels
    channel_pattern = r'<\|start\|>assistant<\|channel\|>([^<]+)<\|message\|>(.*?)(?=<\|end\|>|<\|start\|>assistant<\|channel\|>|$)'
    channels = re.findall(channel_pattern, content, re.DOTALL)
    
    print(f"Found {len(channels)} harmony channels")
    
    # Process channels
    allowed_channels = {'response', 'final', 'answer', 'assistant'}
    suppressed_channels = {'analysis', 'thinking', 'internal', 'reasoning'}
    
    final_content_parts = []
    
    # Add observer text if any
    if observer_text:
        final_content_parts.append(observer_text)
    
    # Process each channel
    for channel_name, channel_content in channels:
        channel_name = channel_name.strip().lower()
        
        if channel_name in allowed_channels:
            # Include content from allowed channels
            clean_content = channel_content.strip()
            if clean_content:
                final_content_parts.append(clean_content)
                print(f"Including content from '{channel_name}' channel")
        elif channel_name in suppressed_channels:
            print(f"Suppressing '{channel_name}' channel content")
        else:
            # For unknown channels, include them to be safe
            clean_content = channel_content.strip()
            if clean_content:
                final_content_parts.append(clean_content)
                print(f"Including content from unknown '{channel_name}' channel")
    
    # Generate final content
    final_content = '\n\n'.join(part for part in final_content_parts if part.strip())
    
    print(f"Harmony filtering complete: {len(content)} → {len(final_content)} chars")
    return final_content

if __name__ == "__main__":
    # Test with the user's reported content (updated format)
    test_content = """Initializing tool analysis...Loading standard RAG tools...Analyzing tool requirements...No dynamic tools needed for this request<|start|>assistant<|channel|>analysis<|message|>The user says: "I wanna build a 3d filament extruder". They want to build a 3D filament extruder. We need to respond with guidance, steps, design considerations, parts list, maybe some instructions. The user likely wants help building an extruder for a 3D printer? Or they want to build a filament extruder that can extrude filament into a nozzle? They might be wanting to build a custom extruder for a 3D printer or a filament extruder for a filament spooler? Let's interpret: "I wanna build a 3d filament extruder". So the user wants instructions on building an extruder. We need to provide a detailed guide, including design considerations, parts list, mechanical design, electronics, firmware, assembly steps, troubleshooting, etc.

We should ask clarifying questions? The user didn't specify details: maybe they want a DIY extruder for a 3D printer. Provide a comprehensive plan. Let's produce an answer with sections: Overview, Design Goals, Parts List, Mechanical Design, Electronics & Firmware, Assembly Steps, Testing & Calibration, Troubleshooting Tips.

We can also include some recommended tools and resources. Also mention potential pitfalls. We might ask if they want to build from scratch or modify existing extruder? But we can give a generic guide.

Let's produce an answer with bullet points, diagrams maybe described textually. Provide step-by-step instructions. Use clear language. Let's do that.<|end|><|start|>assistant<|channel|>final<|message|>## Building Your Own 3‑D Filament Extruder
*(A complete "from‑scratch" guide – you can adapt it to a specific printer or use it as a standalone extruder for filament spooling)*

### 1. What You'll Build

| Item | Purpose |
|------|----------|
| **Extruder body** | Holds the hot end, motor, and drive gear. |
| **Hot‑end assembly** | Heat‑block + nozzle + heater cartridge. |

> **Goal:** A reliable, low‑friction extruder that can push filament at 200–300 mm/s with a max temp of ~250 °C, suitable for PLA or ABS."""
    
    print("Original content:")
    print(repr(test_content[:200]) + "...")
    print("\n" + "="*50 + "\n")
    
    result = filter_harmony_channels_v2(test_content)
    
    print("\nProcessed content:")
    print(repr(result[:200]) + "...")
    print("\nFormatted result:")
    print(result)