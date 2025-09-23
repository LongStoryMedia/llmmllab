#!/usr/bin/env python3
"""
Simple test of harmony channel routing logic without dependencies.
"""

def test_thinking_cleanup():
    """Test that thinking content is properly cleaned of harmony markers."""
    
    # Example content from the console log
    raw_thinking = 'The user says: "I wanna build a 3d filament extruder". They want to build... Ok let\'s produce a thorough answer.<|end|><|start|>assistant'
    
    # Clean up harmony markers
    clean_thinking = raw_thinking.replace('<|end|>', '').replace('<|start|>assistant', '').strip()
    
    print("Raw thinking content:")
    print(repr(raw_thinking))
    print("\nCleaned thinking content:")
    print(repr(clean_thinking))
    print(f"\nContains harmony markers: {'<|' in clean_thinking}")

def test_channel_detection():
    """Test basic channel detection logic."""
    
    # Test content samples
    test_samples = [
        "Regular content without channels",
        "<|channel|>analysis<|message|>This is thinking content",
        "<|channel|>final<|message|>This is final response",
        "Initializing tool analysis...<|channel|>analysis<|message|>Let me analyze this<|end|><|start|>assistant<|channel|>final<|message|>Here's my response",
    ]
    
    print("\nTesting channel detection patterns...")
    
    for i, sample in enumerate(test_samples, 1):
        print(f"\nTest {i}: {repr(sample[:50])}...")
        
        # Check for analysis channel
        has_analysis = '<|channel|>analysis<|message|>' in sample
        print(f"  Has analysis channel: {has_analysis}")
        
        # Check for final channel
        has_final = '<|channel|>final<|message|>' in sample
        print(f"  Has final channel: {has_final}")
        
        # Extract analysis content if present
        if has_analysis and has_final:
            analysis_start_pos = sample.find('<|channel|>analysis<|message|>') + len('<|channel|>analysis<|message|>')
            final_marker_pos = sample.find('<|channel|>final<|message|>')
            
            # Extract raw thinking content
            raw_thinking = sample[analysis_start_pos:final_marker_pos]
            # Clean up harmony markers from thinking content
            clean_thinking = raw_thinking.replace('<|end|>', '').replace('<|start|>assistant', '').strip()
            
            print(f"  Raw analysis: {repr(raw_thinking)}")
            print(f"  Clean analysis: {repr(clean_thinking)}")
        
        # Extract final content if present
        if has_final:
            start_pos = sample.find('<|channel|>final<|message|>') + len('<|channel|>final<|message|>')
            final_content = sample[start_pos:].replace('<|end|>', '').replace('<|return|>', '').strip()
            print(f"  Final content: {repr(final_content[:100])}...")

if __name__ == "__main__":
    test_thinking_cleanup()
    test_channel_detection()