from fastapi import APIRouter
from typing import Optional
from models.openai.deleted_role_assignment_resource import DeletedRoleAssignmentResource
from models.openai.group_role_assignment import GroupRoleAssignment
from models.openai.public_assign_organization_group_role_body import (
    PublicAssignOrganizationGroupRoleBody,
)
from models.openai.public_create_organization_role_body import (
    PublicCreateOrganizationRoleBody,
)
from models.openai.public_role_list_resource import PublicRoleListResource
from models.openai.public_update_organization_role_body import (
    PublicUpdateOrganizationRoleBody,
)
from models.openai.role import Role
from models.openai.role_deleted_resource import RoleDeletedResource
from models.openai.role_list_resource import RoleListResource
from models.openai.user_role_assignment import UserRoleAssignment


router = APIRouter(prefix="/projects", tags=["Projects"])


@router.get("/{project_id}/groups/{group_id}/roles")
async def list_project_group_role_assignments(
    project_id: str, group_id: str
) -> RoleListResource:
    """Operation ID: list-project-group-role-assignments"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{project_id}/groups/{group_id}/roles")
async def assign_project_group_role(
    project_id: str, group_id: str, body: PublicAssignOrganizationGroupRoleBody
) -> GroupRoleAssignment:
    """Operation ID: assign-project-group-role"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.delete("/{project_id}/groups/{group_id}/roles/{role_id}")
async def unassign_project_group_role(
    project_id: str, group_id: str, role_id: str
) -> DeletedRoleAssignmentResource:
    """Operation ID: unassign-project-group-role"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{project_id}/roles")
async def list_project_roles(project_id: str) -> PublicRoleListResource:
    """Operation ID: list-project-roles"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{project_id}/roles")
async def create_project_role(
    project_id: str, body: PublicCreateOrganizationRoleBody
) -> Role:
    """Operation ID: create-project-role"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.delete("/{project_id}/roles/{role_id}")
async def delete_project_role(project_id: str, role_id: str) -> RoleDeletedResource:
    """Operation ID: delete-project-role"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{project_id}/roles/{role_id}")
async def update_project_role(
    project_id: str, role_id: str, body: PublicUpdateOrganizationRoleBody
) -> Role:
    """Operation ID: update-project-role"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{project_id}/users/{user_id}/roles")
async def list_project_user_role_assignments(
    project_id: str, user_id: str
) -> RoleListResource:
    """Operation ID: list-project-user-role-assignments"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{project_id}/users/{user_id}/roles")
async def assign_project_user_role(
    project_id: str, user_id: str, body: PublicAssignOrganizationGroupRoleBody
) -> UserRoleAssignment:
    """Operation ID: assign-project-user-role"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.delete("/{project_id}/users/{user_id}/roles/{role_id}")
async def unassign_project_user_role(
    project_id: str, user_id: str, role_id: str
) -> DeletedRoleAssignmentResource:
    """Operation ID: unassign-project-user-role"""
    raise NotImplementedError("Endpoint not yet implemented")
