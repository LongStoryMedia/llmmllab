#!/usr/bin/env python3
"""
Simple test to see what workflow type gets selected for technical questions.
"""

import asyncio
import json
import requests
import uuid
from typing import List, Dict, Any

from utils.logging import llmmllogger


class WorkflowSelectionTest:
    """Test workflow selection for different question types."""

    def __init__(self):
        self.logger = llmmllogger.logger.bind(component="WorkflowSelectionTest")
        self.base_url = "http://localhost:8000"

    async def test_question(self, question: str) -> None:
        """Test workflow selection for a specific question"""
        self.logger.info(f"📝 Testing question: {question}")
        
        try:
            # Create conversation in database first
            from db import storage
            import os
            
            # Initialize storage if needed
            if not storage.initialized:
                db_host = os.getenv("DB_HOST", "psql-primary.psql.svc.cluster.local")
                db_port = os.getenv("DB_PORT", "5432")
                db_user = os.getenv("DB_USER", "lsm")
                db_password = os.getenv("DB_PASSWORD", "")
                db_name = os.getenv("DB_NAME", "llmmll")
                db_sslmode = os.getenv("DB_SSLMODE", "disable")

                connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode={db_sslmode}"
                await storage.initialize(connection_string)
            
            # Create a conversation for testing
            conversation_id = await storage.get_service(storage.conversation).create_conversation(
                user_id="test_user",
                title=f"Test: {question[:50]}"
            )
            
            if not conversation_id:
                self.logger.error("Failed to create conversation for testing")
                return
            
            # Now test the chat completion - use the correct Message structure
            chat_request = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    }
                ],
                "conversation_id": conversation_id
            }
            
            # Make the chat completion request
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json=chat_request,
                stream=True
            )
            
            if response.status_code != 200:
                self.logger.error(f"❌ Chat completion failed: {response.text}")
                return
            
            # Process streaming response
            workflow_detected = []
            response_chunks = []
            line_count = 0
            
            for line in response.iter_lines():
                if line:
                    line_count += 1
                    line_str = line.decode('utf-8')
                    # Since the format is not Server-Sent Events, parse each line as JSON
                    try:
                        data = json.loads(line_str)
                        
                        # Look for message content
                        if 'message' in data and data['message']:
                            message = data['message']
                            if 'content' in message and message['content']:
                                for content_item in message['content']:
                                    if 'text' in content_item:
                                        text = content_item['text']
                                        response_chunks.append(text)
                                        # Look for workflow-related content
                                        if any(keyword in text.lower() for keyword in 
                                              ['engineering', 'workflow', 'routing', 'classifier', 'agent']):
                                            workflow_detected.append(text)
                        
                        # Look for analyses in the response
                        if 'analyses' in data and data['analyses']:
                            for analysis in data['analyses']:
                                if 'workflow_type' in analysis:
                                    workflow_type = analysis['workflow_type']
                                    self.logger.info(f"🎯 Detected workflow type: {workflow_type}")
                                    workflow_detected.append(f"workflow_type: {workflow_type}")
                        
                        # Check for stream end
                        if data.get('type') == 'stream_end':
                            self.logger.debug("Stream ended")
                            break
                            
                    except json.JSONDecodeError as e:
                        self.logger.debug(f"Failed to parse JSON: {e} - Raw: {line_str[:100]}")
                        continue
                        
            self.logger.info(f"Processed {line_count} total lines from stream")
            
            # Analyze the complete response
            full_response = ''.join(response_chunks)
            self.logger.info(f"📊 Question: {question}")
            self.logger.info(f"🎯 Workflow indicators: {workflow_detected if workflow_detected else 'None detected'}")
            self.logger.info(f"📝 Response length: {len(full_response)} chars")
            if full_response:
                # Look for any workflow-related content in the full response
                workflow_terms = ['engineering', 'workflow', 'routing', 'classifier', 'agent']
                found_terms = [term for term in workflow_terms if term.lower() in full_response.lower()]
                if found_terms:
                    self.logger.info(f"🔍 Found workflow terms in response: {found_terms}")
                else:
                    self.logger.info("❓ No workflow terms found in full response")
                    
                # Show first and last parts of response to see structure
                if len(full_response) > 400:
                    self.logger.info(f"📄 Response start: {full_response[:200]}")
                    self.logger.info(f"📄 Response end: {full_response[-200:]}")
                else:
                    self.logger.info(f"📄 Full response: {full_response}")
            else:
                self.logger.warning("⚠️  Empty response received")
            self.logger.info("─" * 80)
                
        except Exception as e:
            self.logger.error(f"❌ Error testing question '{question}': {e}")

    async def test_technical_question_workflow_selection(self):
        """Test various technical questions to see which workflows get selected."""
        self.logger.info("🎯 Testing workflow selection for technical questions")
        
        technical_questions = [
            "Explain how quicksort algorithm works step by step"
            # Only test one question for now to get detailed output
            # "How do I implement a binary search tree in Python?",
            # "What are the design patterns for microservices architecture?", 
            # "Show me how to build a REST API with authentication",
            # "Explain the difference between SQL and NoSQL databases"
        ]
        
        for question in technical_questions:
            await self.test_question(question)
            await asyncio.sleep(1)  # Brief pause between tests

    async def run_all_tests(self):
        """Run all workflow selection tests."""
        self.logger.info("🚀 Starting workflow selection investigation")
        
        try:
            await self.test_technical_question_workflow_selection()
            self.logger.info("✅ All workflow selection tests completed")
            
        except Exception as e:
            self.logger.error(f"❌ Test execution failed: {e}")
            raise


async def main():
    """Main test runner."""
    test = WorkflowSelectionTest()
    await test.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())