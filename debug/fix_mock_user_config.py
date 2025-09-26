#!/usr/bin/env python3

import re

# Read the file
with open('test_real_end_to_end_pipeline.py', 'r') as f:
    content = f.read()

# Fix the MockUserConfig class
old_pattern = r'class MockUserConfig:\s+def __init__\(self\):\s+self\.user_id = "test_dedup_user"\s+self\.model_profiles.*?\n\s+# Initialize deduplicator'

new_code = '''class MockUserConfig:
                def __init__(self):
                    self.user_id = "test_dedup_user"
                    self.model_profiles = type("MockModelProfiles", (), {"embedding_profile_id": None})()
            
            # Initialize deduplicator'''

content = re.sub(old_pattern, new_code, content, flags=re.DOTALL)

# Write the fixed content back
with open('test_real_end_to_end_pipeline.py', 'w') as f:
    f.write(content)

print("Fixed MockUserConfig class")