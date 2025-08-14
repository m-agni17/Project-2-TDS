#!/usr/bin/env python3
"""
Test script for the Generic Data Analyst Agent API
Demonstrates how to use the API with different types of websites and questions.
"""

import requests
import json
import os

def test_api_endpoint(questions_file: str, description: str):
    """Test the API with a specific questions file."""
    
    print(f"\n🧪 Testing: {description}")
    print(f"Questions file: {questions_file}")
    print("-" * 50)
    
    # Check if questions file exists
    if not os.path.exists(questions_file):
        print(f"❌ Questions file {questions_file} not found!")
        return
    
    # Make API request
    try:
        with open(questions_file, 'rb') as f:
            files = [('files', ('questions.txt', f, 'text/plain'))]
            
            response = requests.post(
                "http://localhost:8000/api/",
                files=files,
                timeout=180  # 3 minute timeout
            )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"Response type: {type(result)}")
            print(f"Number of answers: {len(result) if isinstance(result, list) else 'N/A'}")
            
            # Print answers (truncated)
            if isinstance(result, list):
                for i, answer in enumerate(result, 1):
                    answer_str = str(answer)
                    if len(answer_str) > 100:
                        answer_str = answer_str[:100] + "..."
                    print(f"Answer {i}: {answer_str}")
            else:
                print(f"Result: {str(result)[:200]}...")
                
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (>3 minutes)")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Is the server running?")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def main():
    """Run API tests."""
    
    print("🚀 Generic Data Analyst Agent - API Test")
    print("=" * 60)
    
    # Check if server is running
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ Server is running")
            print(f"Status: {health_data.get('status')}")
            print(f"Max file size: {health_data.get('max_file_size_mb')}MB")
            print(f"Timeout: {health_data.get('response_timeout_seconds')}s")
        else:
            print("❌ Server health check failed")
            return
    except:
        print("❌ Server is not running! Please start with: python main.py")
        return
    
    # Test cases
    test_cases = [
        ("sample_questions.txt", "Original test: Highest grossing films"),
    ]
    
    for questions_file, description in test_cases:
        test_api_endpoint(questions_file, description)
    
    print("\n" + "=" * 60)
    print("🏁 Testing complete!")
    print("\nTo create your own test:")
    print("1. Create a new questions.txt file")
    print("2. Add any website URL and questions about the content")
    print("3. Run: curl -X POST 'http://localhost:8000/api/' -F 'questions.txt=@your_questions.txt'")

if __name__ == "__main__":
    main() 