from fastapi import APIRouter
from typing import Optional
from models.openai.create_message_request import CreateMessageRequest
from models.openai.create_run_request import CreateRunRequest
from models.openai.create_thread_and_run_request import CreateThreadAndRunRequest
from models.openai.create_thread_request import CreateThreadRequest
from models.openai.delete_message_response import DeleteMessageResponse
from models.openai.delete_thread_response import DeleteThreadResponse
from models.openai.list_messages_response import ListMessagesResponse
from models.openai.list_run_steps_response import ListRunStepsResponse
from models.openai.list_runs_response import ListRunsResponse
from models.openai.message_object import MessageObject
from models.openai.modify_message_request import ModifyMessageRequest
from models.openai.modify_run_request import ModifyRunRequest
from models.openai.modify_thread_request import ModifyThreadRequest
from models.openai.run_object import RunObject
from models.openai.run_step_object import RunStepObject
from models.openai.submit_tool_outputs_run_request import SubmitToolOutputsRunRequest
from models.openai.thread_object import ThreadObject


router = APIRouter(prefix="/threads", tags=["Threads"])


@router.post("/")
async def createThread(body: CreateThreadRequest) -> ThreadObject:
    """Operation ID: createThread"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/runs")
async def createThreadAndRun(body: CreateThreadAndRunRequest) -> RunObject:
    """Operation ID: createThreadAndRun"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.delete("/{thread_id}")
async def deleteThread(thread_id: str) -> DeleteThreadResponse:
    """Operation ID: deleteThread"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{thread_id}")
async def getThread(thread_id: str) -> ThreadObject:
    """Operation ID: getThread"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{thread_id}")
async def modifyThread(thread_id: str, body: ModifyThreadRequest) -> ThreadObject:
    """Operation ID: modifyThread"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{thread_id}/messages")
async def listMessages(thread_id: str) -> ListMessagesResponse:
    """Operation ID: listMessages"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{thread_id}/messages")
async def createMessage(thread_id: str, body: CreateMessageRequest) -> MessageObject:
    """Operation ID: createMessage"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.delete("/{thread_id}/messages/{message_id}")
async def deleteMessage(thread_id: str, message_id: str) -> DeleteMessageResponse:
    """Operation ID: deleteMessage"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{thread_id}/messages/{message_id}")
async def getMessage(thread_id: str, message_id: str) -> MessageObject:
    """Operation ID: getMessage"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{thread_id}/messages/{message_id}")
async def modifyMessage(
    thread_id: str, message_id: str, body: ModifyMessageRequest
) -> MessageObject:
    """Operation ID: modifyMessage"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{thread_id}/runs")
async def listRuns(thread_id: str) -> ListRunsResponse:
    """Operation ID: listRuns"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{thread_id}/runs")
async def createRun(thread_id: str, body: CreateRunRequest) -> RunObject:
    """Operation ID: createRun"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{thread_id}/runs/{run_id}")
async def getRun(thread_id: str, run_id: str) -> RunObject:
    """Operation ID: getRun"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{thread_id}/runs/{run_id}")
async def modifyRun(thread_id: str, run_id: str, body: ModifyRunRequest) -> RunObject:
    """Operation ID: modifyRun"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{thread_id}/runs/{run_id}/cancel")
async def cancelRun(thread_id: str, run_id: str) -> RunObject:
    """Operation ID: cancelRun"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{thread_id}/runs/{run_id}/steps")
async def listRunSteps(thread_id: str, run_id: str) -> ListRunStepsResponse:
    """Operation ID: listRunSteps"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{thread_id}/runs/{run_id}/steps/{step_id}")
async def getRunStep(thread_id: str, run_id: str, step_id: str) -> RunStepObject:
    """Operation ID: getRunStep"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{thread_id}/runs/{run_id}/submit_tool_outputs")
async def submitToolOuputsToRun(
    thread_id: str, run_id: str, body: SubmitToolOutputsRunRequest
) -> RunObject:
    """Operation ID: submitToolOuputsToRun"""
    raise NotImplementedError("Endpoint not yet implemented")
