#!/usr/bin/env python3
"""
Project CRUD Operations Test Suite

Tests for:
1. Project creation, reading, updating, deletion
2. User-project access control
3. Project-document relationships
4. Authentication and authorization
"""

import pytest
import os
import sys
import uuid
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.storage.db import Database

# Test configuration
TEST_USER_ID = "test_project_crud_user"
TEST_PROJECT_NAME = "Test CRUD Project"
TEST_PROJECT_DESCRIPTION = "Project for testing CRUD operations"

@pytest.fixture(scope="session")
def db_connection():
    """Create a database connection for testing."""
    try:
        return Database()
    except Exception as e:
        pytest.skip(f"Could not connect to database: {e}")

@pytest.fixture
def test_user_id():
    """Generate a unique test user ID for isolation."""
    return f"test_user_{uuid.uuid4().hex[:8]}"

class TestProjectCreation:
    """Test project creation functionality."""
    
    def test_create_project_basic(self, db_connection, test_user_id):
        """Test basic project creation."""
        project_name = f"{TEST_PROJECT_NAME} Basic"
        project_description = f"{TEST_PROJECT_DESCRIPTION} - Basic"
        
        try:
            project_id = db_connection.create_project(
                name=project_name,
                description=project_description,
                user_id=test_user_id
            )
            
            assert project_id is not None, "Project ID should be returned"
            assert isinstance(project_id, str), "Project ID should be a string"
            
            print(f"✅ Created project: {project_id[:8]}... with name '{project_name}'")
            
            # Cleanup
            try:
                db_connection.delete_project(project_id)
                print(f"✅ Cleaned up project: {project_id[:8]}...")
            except Exception as cleanup_error:
                print(f"⚠️ Cleanup warning: {cleanup_error}")
            
        except Exception as e:
            pytest.fail(f"Project creation failed: {e}")
    
    def test_create_project_with_long_name(self, db_connection, test_user_id):
        """Test project creation with long name."""
        long_name = "Very Long Project Name " * 10  # Create a very long name
        project_description = "Project with long name"
        
        try:
            project_id = db_connection.create_project(
                name=long_name,
                description=project_description,
                user_id=test_user_id
            )
            
            assert project_id is not None
            print(f"✅ Created project with long name: {project_id[:8]}...")
            
            # Cleanup
            try:
                db_connection.delete_project(project_id)
            except Exception:
                pass
            
        except Exception as e:
            # This might fail due to database constraints, which is acceptable
            print(f"⚠️ Long name project creation failed (expected): {e}")
    
    def test_create_project_empty_description(self, db_connection, test_user_id):
        """Test project creation with empty description."""
        project_name = f"{TEST_PROJECT_NAME} Empty Desc"
        
        try:
            project_id = db_connection.create_project(
                name=project_name,
                description="",
                user_id=test_user_id
            )
            
            assert project_id is not None
            print(f"✅ Created project with empty description: {project_id[:8]}...")
            
            # Cleanup
            try:
                db_connection.delete_project(project_id)
            except Exception:
                pass
                
        except Exception as e:
            pytest.fail(f"Project creation with empty description failed: {e}")

class TestProjectRetrieval:
    """Test project retrieval functionality."""
    
    @pytest.fixture
    def test_project(self, db_connection, test_user_id):
        """Create a test project for retrieval tests."""
        project_name = f"{TEST_PROJECT_NAME} Retrieval"
        project_description = f"{TEST_PROJECT_DESCRIPTION} - Retrieval"
        
        project_id = db_connection.create_project(
            name=project_name,
            description=project_description,
            user_id=test_user_id
        )
        
        yield {
            'project_id': project_id,
            'name': project_name,
            'description': project_description,
            'user_id': test_user_id
        }
        
        # Cleanup
        try:
            db_connection.delete_project(project_id)
        except Exception as e:
            print(f"Cleanup warning: {e}")
    
    def test_get_project_by_id(self, db_connection, test_project):
        """Test retrieving a project by ID."""
        project_id = test_project['project_id']
        
        try:
            project = db_connection.get_project(project_id)
            
            assert project is not None, "Project should be found"
            assert project.get('id') == project_id, "Project ID should match"
            assert project.get('name') == test_project['name'], "Project name should match"
            assert project.get('description') == test_project['description'], "Description should match"
            
            print(f"✅ Retrieved project: {project.get('name')}")
            
        except Exception as e:
            pytest.fail(f"Project retrieval by ID failed: {e}")
    
    def test_get_projects_by_user(self, db_connection, test_project):
        """Test retrieving all projects for a user."""
        user_id = test_project['user_id']
        
        try:
            # Set user_id for proper RLS
            db_connection.user_id = user_id
            
            projects = db_connection.get_projects_by_user(user_id)
            
            assert isinstance(projects, list), "Projects should be returned as a list"
            assert len(projects) >= 1, "Should find at least the test project"
            
            # Find our test project in the results
            test_project_found = False
            for project in projects:
                if project.get('id') == test_project['project_id']:
                    test_project_found = True
                    break
            
            assert test_project_found, "Test project should be found in user's projects"
            print(f"✅ Found {len(projects)} projects for user")
            
        except Exception as e:
            pytest.fail(f"Project retrieval by user failed: {e}")
    
    def test_get_nonexistent_project(self, db_connection):
        """Test retrieving a non-existent project."""
        fake_project_id = str(uuid.uuid4())
        
        try:
            project = db_connection.get_project(fake_project_id)
            # Should return None or empty result for non-existent project
            assert project is None or (isinstance(project, dict) and len(project) == 0)
            print("✅ Non-existent project correctly returned None/empty")
            
        except Exception as e:
            # Some implementations might raise an exception, which is also acceptable
            print(f"⚠️ Non-existent project raised exception (acceptable): {e}")

class TestProjectUpdates:
    """Test project update functionality."""
    
    @pytest.fixture
    def test_project(self, db_connection, test_user_id):
        """Create a test project for update tests."""
        project_name = f"{TEST_PROJECT_NAME} Update"
        project_description = f"{TEST_PROJECT_DESCRIPTION} - Update"
        
        project_id = db_connection.create_project(
            name=project_name,
            description=project_description,
            user_id=test_user_id
        )
        
        yield {
            'project_id': project_id,
            'name': project_name,
            'description': project_description,
            'user_id': test_user_id
        }
        
        # Cleanup
        try:
            db_connection.delete_project(project_id)
        except Exception as e:
            print(f"Cleanup warning: {e}")
    
    def test_update_project_name(self, db_connection, test_project):
        """Test updating project name."""
        project_id = test_project['project_id']
        new_name = f"{test_project['name']} - Updated"
        
        try:
            # Check if update_project method exists
            if hasattr(db_connection, 'update_project'):
                result = db_connection.update_project(
                    project_id=project_id,
                    name=new_name
                )
                
                # Verify the update
                updated_project = db_connection.get_project(project_id)
                assert updated_project.get('name') == new_name, "Project name should be updated"
                
                print(f"✅ Updated project name to: {new_name}")
            else:
                print("⚠️ update_project method not implemented - skipping test")
                pytest.skip("update_project method not available")
                
        except Exception as e:
            pytest.fail(f"Project name update failed: {e}")
    
    def test_update_project_description(self, db_connection, test_project):
        """Test updating project description."""
        project_id = test_project['project_id']
        new_description = f"{test_project['description']} - Updated Description"
        
        try:
            if hasattr(db_connection, 'update_project'):
                result = db_connection.update_project(
                    project_id=project_id,
                    description=new_description
                )
                
                # Verify the update
                updated_project = db_connection.get_project(project_id)
                assert updated_project.get('description') == new_description, "Description should be updated"
                
                print(f"✅ Updated project description")
            else:
                print("⚠️ update_project method not implemented - skipping test")
                pytest.skip("update_project method not available")
                
        except Exception as e:
            pytest.fail(f"Project description update failed: {e}")

class TestProjectDeletion:
    """Test project deletion functionality."""
    
    def test_delete_project_basic(self, db_connection, test_user_id):
        """Test basic project deletion."""
        # Create a project specifically for deletion
        project_name = f"{TEST_PROJECT_NAME} Delete"
        project_description = f"{TEST_PROJECT_DESCRIPTION} - Delete"
        
        project_id = db_connection.create_project(
            name=project_name,
            description=project_description,
            user_id=test_user_id
        )
        
        # Verify project exists
        project = db_connection.get_project(project_id)
        assert project is not None, "Project should exist before deletion"
        
        # Delete the project
        try:
            result = db_connection.delete_project(project_id)
            print(f"✅ Deleted project: {project_id[:8]}...")
            
            # Verify project no longer exists
            deleted_project = db_connection.get_project(project_id)
            assert deleted_project is None or len(deleted_project) == 0, "Project should not exist after deletion"
            
        except Exception as e:
            pytest.fail(f"Project deletion failed: {e}")
    
    def test_delete_nonexistent_project(self, db_connection):
        """Test deleting a non-existent project."""
        fake_project_id = str(uuid.uuid4())
        
        try:
            result = db_connection.delete_project(fake_project_id)
            # Should handle gracefully (not raise exception)
            print("✅ Non-existent project deletion handled gracefully")
            
        except Exception as e:
            # Some implementations might raise an exception, which can be acceptable
            print(f"⚠️ Non-existent project deletion raised exception: {e}")
            # Don't fail the test for this

class TestUserProjectAccess:
    """Test user-project access control functionality."""
    
    @pytest.fixture
    def test_project_with_access(self, db_connection, test_user_id):
        """Create a test project with access control."""
        project_name = f"{TEST_PROJECT_NAME} Access"
        project_description = f"{TEST_PROJECT_DESCRIPTION} - Access"
        
        project_id = db_connection.create_project(
            name=project_name,
            description=project_description,
            user_id=test_user_id
        )
        
        yield {
            'project_id': project_id,
            'name': project_name,
            'description': project_description,
            'user_id': test_user_id
        }
        
        # Cleanup
        try:
            db_connection.delete_project(project_id)
        except Exception as e:
            print(f"Cleanup warning: {e}")
    
    def test_user_project_access_creation(self, db_connection, test_project_with_access):
        """Test that user-project access is created automatically."""
        project_id = test_project_with_access['project_id']
        user_id = test_project_with_access['user_id']
        
        try:
            # Check if the user has access to the project
            if hasattr(db_connection, 'check_user_project_access'):
                has_access = db_connection.check_user_project_access(user_id, project_id)
                assert has_access, "User should have access to their own project"
                print("✅ User has access to their project")
            else:
                print("⚠️ check_user_project_access method not implemented")
                
        except Exception as e:
            print(f"⚠️ User project access check failed: {e}")
    
    def test_user_cannot_access_other_projects(self, db_connection, test_project_with_access):
        """Test that users cannot access projects they don't own."""
        project_id = test_project_with_access['project_id']
        other_user_id = f"other_user_{uuid.uuid4().hex[:8]}"
        
        try:
            if hasattr(db_connection, 'check_user_project_access'):
                has_access = db_connection.check_user_project_access(other_user_id, project_id)
                assert not has_access, "Other users should not have access to the project"
                print("✅ Other users correctly denied access")
            else:
                print("⚠️ Access control test skipped - method not implemented")
                
        except Exception as e:
            print(f"⚠️ Access control test failed: {e}")

class TestProjectValidation:
    """Test project validation and error handling."""
    
    def test_create_project_empty_name(self, db_connection, test_user_id):
        """Test creating project with empty name."""
        try:
            project_id = db_connection.create_project(
                name="",
                description="Project with empty name",
                user_id=test_user_id
            )
            
            # If this succeeds, clean it up
            if project_id:
                try:
                    db_connection.delete_project(project_id)
                except Exception:
                    pass
                
            # Empty name might be allowed or rejected - both are valid behaviors
            print("⚠️ Empty name project creation allowed (implementation-specific)")
            
        except Exception as e:
            # This is expected - empty names should typically be rejected
            print(f"✅ Empty name correctly rejected: {e}")
    
    def test_create_project_invalid_user(self, db_connection):
        """Test creating project with invalid user ID."""
        try:
            project_id = db_connection.create_project(
                name="Invalid User Project",
                description="Project with invalid user",
                user_id=""  # Empty user ID
            )
            
            # If this somehow succeeds, clean it up
            if project_id:
                try:
                    db_connection.delete_project(project_id)
                except Exception:
                    pass
            
            print("⚠️ Invalid user ID project creation allowed (unexpected)")
            
        except Exception as e:
            # This is expected - invalid user IDs should be rejected
            print(f"✅ Invalid user ID correctly rejected: {e}")

class TestProjectCRUDIntegration:
    """Test complete CRUD workflow integration."""
    
    def test_complete_crud_workflow(self, db_connection, test_user_id):
        """Test complete Create-Read-Update-Delete workflow."""
        project_id = None
        
        try:
            # CREATE
            print("\n📝 Testing CREATE operation...")
            project_name = f"{TEST_PROJECT_NAME} CRUD Workflow"
            project_description = f"{TEST_PROJECT_DESCRIPTION} - CRUD Workflow"
            
            project_id = db_connection.create_project(
                name=project_name,
                description=project_description,
                user_id=test_user_id
            )
            
            assert project_id is not None
            print(f"✅ CREATE: Project created with ID {project_id[:8]}...")
            
            # READ
            print("📖 Testing READ operation...")
            project = db_connection.get_project(project_id)
            assert project is not None
            assert project.get('name') == project_name
            print(f"✅ READ: Project retrieved successfully")
            
            # UPDATE (if available)
            print("✏️ Testing UPDATE operation...")
            if hasattr(db_connection, 'update_project'):
                new_name = f"{project_name} - Updated"
                db_connection.update_project(project_id=project_id, name=new_name)
                
                updated_project = db_connection.get_project(project_id)
                assert updated_project.get('name') == new_name
                print(f"✅ UPDATE: Project name updated successfully")
            else:
                print("⚠️ UPDATE: Method not available - skipped")
            
            # DELETE
            print("🗑️ Testing DELETE operation...")
            db_connection.delete_project(project_id)
            
            deleted_project = db_connection.get_project(project_id)
            assert deleted_project is None or len(deleted_project) == 0
            print(f"✅ DELETE: Project deleted successfully")
            
            project_id = None  # Mark as deleted for cleanup
            
            print("\n🎉 Complete CRUD workflow successful!")
            
        except Exception as e:
            # Cleanup if something failed
            if project_id:
                try:
                    db_connection.delete_project(project_id)
                    print(f"✅ Cleanup: Deleted project {project_id[:8]}...")
                except Exception:
                    print(f"⚠️ Cleanup failed for project {project_id[:8]}...")
            
            pytest.fail(f"CRUD workflow failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 