#!/usr/bin/env python3
"""
Enhanced Legal AI System - Comprehensive Test Runner
Executes all test suites with detailed reporting and validation
"""

import os
import sys
import asyncio
import subprocess
import json
from datetime import datetime
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class EnhancedTestRunner:
    """Comprehensive test runner for the Enhanced Legal AI System"""
    
    def __init__(self):
        self.test_root = Path(__file__).parent
        self.results = {
            'unit_tests': {},
            'integration_tests': {},
            'end_to_end_tests': {},
            'summary': {}
        }
        
        # Test suite definitions with new structure
        self.test_suites = {
            'unit': {
                "API Cost Calculation": "unit/test_api_costs.py",
                "PDF Extractor API Logging": "unit/test_pdf_extractor_api_logging.py"
            },
            'integration': {
                "Database Functionality": "integration/test_database_functionality.py",
                "Project CRUD Operations": "integration/test_project_crud.py",
                "Food Industry Integration": "integration/test_food_industry_integration.py",
                "LightRAG Integration": "integration/test_lightrag_integration.py"
            },
            'end_to_end': {
                "Document Processing Flow": "end_to_end/test_document_processing_flow.py",
                "Enhanced System Integration": "end_to_end/test_enhanced_agricultural_system.py",
                "Enhanced System Validation": "end_to_end/test_enhanced_system.py"
            }
        }
        
    def print_header(self):
        """Print test runner header"""
        print("=" * 80)
        print("🧪 ENHANCED LEGAL AI SYSTEM - COMPREHENSIVE TEST RUNNER")
        print("=" * 80)
        print(f"📅 Test Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Test Directory: {self.test_root}")
        print(f"🐍 Python Version: {sys.version}")
        print("=" * 80)
    
    def run_test_suite(self, suite_name: str, test_files: dict) -> dict:
        """Run a specific test suite"""
        print(f"\n🔄 RUNNING {suite_name.upper()} TESTS")
        print("-" * 60)
        
        suite_results = {}
        
        for test_name, test_file in test_files.items():
            print(f"\n📋 {test_name}")
            print(f"📄 File: {test_file}")
            
            test_path = self.test_root / test_file
            
            if not test_path.exists():
                print(f"❌ Test file not found: {test_path}")
                suite_results[test_name] = {
                    'status': 'FAILED',
                    'error': 'Test file not found',
                    'duration': 0
                }
                continue
            
            try:
                start_time = datetime.now()
                
                # Run pytest with verbose output
                result = subprocess.run([
                    sys.executable, '-m', 'pytest', str(test_path), '-v', '--tb=short'
                ], capture_output=True, text=True, cwd=str(project_root))
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                if result.returncode == 0:
                    print(f"✅ PASSED ({duration:.2f}s)")
                    suite_results[test_name] = {
                        'status': 'PASSED',
                        'duration': duration,
                        'output': result.stdout
                    }
                else:
                    print(f"❌ FAILED ({duration:.2f}s)")
                    print(f"Error: {result.stderr}")
                    suite_results[test_name] = {
                        'status': 'FAILED',
                        'duration': duration,
                        'error': result.stderr,
                        'output': result.stdout
                    }
                    
            except Exception as e:
                print(f"❌ EXCEPTION: {str(e)}")
                suite_results[test_name] = {
                    'status': 'EXCEPTION',
                    'error': str(e),
                    'duration': 0
                }
        
        return suite_results
    
    def run_all_tests(self):
        """Run all test suites"""
        self.print_header()
        
        total_start_time = datetime.now()
        
        # Run each test suite
        for suite_type, test_files in self.test_suites.items():
            suite_results = self.run_test_suite(suite_type, test_files)
            self.results[f'{suite_type}_tests'] = suite_results
        
        total_end_time = datetime.now()
        total_duration = (total_end_time - total_start_time).total_seconds()
        
        # Generate summary
        self.generate_summary(total_duration)
        
        # Print final results
        self.print_final_results()
        
        return self.results
    
    def generate_summary(self, total_duration: float):
        """Generate test execution summary"""
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for suite_type in ['unit_tests', 'integration_tests', 'end_to_end_tests']:
            suite_results = self.results.get(suite_type, {})
            for test_name, result in suite_results.items():
                total_tests += 1
                if result['status'] == 'PASSED':
                    passed_tests += 1
                else:
                    failed_tests += 1
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        self.results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': success_rate,
            'total_duration': total_duration,
            'execution_date': datetime.now().isoformat()
        }
    
    def print_final_results(self):
        """Print final test results summary"""
        summary = self.results['summary']
        
        print("\n" + "=" * 80)
        print("📊 FINAL TEST RESULTS SUMMARY")
        print("=" * 80)
        
        print(f"📈 Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed_tests']}")
        print(f"❌ Failed: {summary['failed_tests']}")
        print(f"📊 Success Rate: {summary['success_rate']:.1f}%")
        print(f"⏱️  Total Duration: {summary['total_duration']:.2f} seconds")
        
        # Print detailed results by category
        print("\n📋 DETAILED RESULTS BY CATEGORY:")
        print("-" * 50)
        
        for suite_type in ['unit_tests', 'integration_tests', 'end_to_end_tests']:
            suite_results = self.results.get(suite_type, {})
            if suite_results:
                print(f"\n🔍 {suite_type.replace('_', ' ').title()}:")
                for test_name, result in suite_results.items():
                    status_icon = "✅" if result['status'] == 'PASSED' else "❌"
                    duration = result.get('duration', 0)
                    print(f"  {status_icon} {test_name} ({duration:.2f}s)")
        
        # Overall status
        if summary['success_rate'] == 100:
            print("\n🎉 ALL TESTS PASSED! System is ready for deployment.")
        elif summary['success_rate'] >= 80:
            print(f"\n⚠️  {summary['success_rate']:.1f}% tests passed. Review failed tests before deployment.")
        else:
            print(f"\n🚨 Only {summary['success_rate']:.1f}% tests passed. System needs attention before deployment.")
        
        print("=" * 80)
    
    def save_results(self, output_file: str = None):
        """Save test results to JSON file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"test_results_{timestamp}.json"
        
        output_path = self.test_root / output_file
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"📄 Test results saved to: {output_path}")

def main():
    """Main function to run all tests"""
    runner = EnhancedTestRunner()
    
    try:
        results = runner.run_all_tests()
        runner.save_results()
        
        # Exit with appropriate code
        summary = results['summary']
        if summary['success_rate'] == 100:
            sys.exit(0)  # All tests passed
        else:
            sys.exit(1)  # Some tests failed
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n💥 Test runner error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 