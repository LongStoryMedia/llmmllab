#!/usr/bin/env python3
"""
Debug System Prompt Content
Identify what's causing massive system prompts in the chat workflow.
"""

import asyncio
import os
import sys
sys.path.append('/app')

from models.user_config import UserConfig
from models.model_profile import ModelProfile, ModelParameters, GPUConfig
from utils.model_profile import ModelProfileStorage
from composer.agents.base_agent import BaseAgent
from composer.config import config
from utils.logging import llmmllogger

logger = llmmllogger.logger.bind(component="debug_system_prompt")

async def analyze_system_prompt():
    """Analyze what's being put into system prompts."""
    
    logger.info("🔍 Starting system prompt analysis...")
    
    # Get storage
    storage = ModelProfileStorage()
    
    # Create a test profile similar to what's used in the E2E test
    profile = ModelProfile(
        user_id="debug_user",
        name="Debug Profile",
        description="Debug profile for system prompt analysis",
        model_name="qwen3-30b-a3b-q4-k-m",
        parameters=ModelParameters(
            num_ctx=100000,
            temperature=0.7,
            max_tokens=4000,
            top_p=0.9,
        ),
        system_prompt="You are a helpful AI assistant with access to web search tools.\nKnowledge cutoff: 2024-04-01",
        gpu_config=GPUConfig(
            no_kv_offload=False,
            gpu_layers=-1,
            main_gpu=-1,
            split_mode="layer",
            offload_kqv=True
        )
    )
    
    # Create agent
    agent = BaseAgent(
        profile=profile,
        component="debug_agent"
    )
    
    # Test with empty conversation (baseline)
    logger.info("📊 Testing with empty conversation...")
    system_prompt_empty = agent._get_system_prompt([])
    logger.info(f"   Empty conversation system prompt: {len(system_prompt_empty):,} characters")
    if len(system_prompt_empty) > 1000:
        logger.warning(f"   Preview: {system_prompt_empty[:500]}...")
    
    # Test with a single message
    logger.info("📊 Testing with single message...")
    simple_messages = [{"role": "user", "content": "Hello, how are you?"}]
    system_prompt_simple = agent._get_system_prompt(simple_messages)
    logger.info(f"   Single message system prompt: {len(system_prompt_simple):,} characters")
    if len(system_prompt_simple) > 1000:
        logger.warning(f"   Preview: {system_prompt_simple[:500]}...")
    
    # Test with messages that might have system content
    logger.info("📊 Testing with system message in conversation...")
    mixed_messages = [
        {"role": "system", "content": "You are analyzing artificial intelligence developments."},
        {"role": "user", "content": "Tell me about recent AI advances."},
        {"role": "assistant", "content": "Recent AI developments include improvements in large language models..."},
        {"role": "user", "content": "What about machine learning advances?"}
    ]
    system_prompt_mixed = agent._get_system_prompt(mixed_messages)
    logger.info(f"   Mixed messages system prompt: {len(system_prompt_mixed):,} characters")
    if len(system_prompt_mixed) > 1000:
        logger.warning(f"   Preview: {system_prompt_mixed[:500]}...")
    
    # Test with a very large system message (like what might come from summarization)
    logger.info("📊 Testing with large system message...")
    large_system_content = "RESEARCH SUMMARY: " + "This is a detailed research summary about artificial intelligence developments. " * 5000  # ~350K chars
    large_messages = [
        {"role": "system", "content": large_system_content},
        {"role": "user", "content": "I need current information about the latest developments in artificial intelligence."}
    ]
    system_prompt_large = agent._get_system_prompt(large_messages)
    logger.info(f"   Large system message system prompt: {len(system_prompt_large):,} characters")
    if len(system_prompt_large) > 1000:
        logger.warning(f"   Preview: {system_prompt_large[:500]}...")
    
    # Estimate tokens for each
    logger.info("📊 Token estimates (4 chars per token):")
    logger.info(f"   Empty: ~{len(system_prompt_empty) // 4:,} tokens")
    logger.info(f"   Simple: ~{len(system_prompt_simple) // 4:,} tokens")
    logger.info(f"   Mixed: ~{len(system_prompt_mixed) // 4:,} tokens")
    logger.info(f"   Large: ~{len(system_prompt_large) // 4:,} tokens")
    
    # Check if large system message replicates our error
    if len(system_prompt_large) // 4 > 40000:
        logger.error(f"🚨 FOUND ISSUE: Large system message creates {len(system_prompt_large) // 4:,} tokens!")
        logger.error("   This would exceed context windows and cause the error we're seeing.")
        
        # Analyze components
        profile_prompt_len = len(profile.system_prompt)
        large_content_len = len(large_system_content)
        logger.info(f"   Profile system prompt: {profile_prompt_len:,} chars")
        logger.info(f"   Large system content: {large_content_len:,} chars")
        logger.info(f"   Combined total: {len(system_prompt_large):,} chars")
    
    logger.info("✅ System prompt analysis complete")

if __name__ == "__main__":
    asyncio.run(analyze_system_prompt())