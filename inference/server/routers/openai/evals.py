from fastapi import APIRouter
from typing import Optional
from models.openai.create_eval_request import CreateEvalRequest
from models.openai.create_eval_run_request import CreateEvalRunRequest
from models.openai.eval import Eval
from models.openai.eval_list import EvalList
from models.openai.eval_run import EvalRun
from models.openai.eval_run_list import EvalRunList
from models.openai.eval_run_output_item import EvalRunOutputItem
from models.openai.eval_run_output_item_list import EvalRunOutputItemList


router = APIRouter(prefix="/evals", tags=["Evals"])


@router.get("/")
async def listEvals() -> EvalList:
    """Operation ID: listEvals"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/")
async def createEval(body: CreateEvalRequest) -> Eval:
    """Operation ID: createEval"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.delete("/{eval_id}")
async def deleteEval(eval_id: str) -> dict:
    """Operation ID: deleteEval"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{eval_id}")
async def getEval(eval_id: str) -> Eval:
    """Operation ID: getEval"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{eval_id}")
async def updateEval(eval_id: str) -> Eval:
    """Operation ID: updateEval"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{eval_id}/runs")
async def getEvalRuns(eval_id: str) -> EvalRunList:
    """Operation ID: getEvalRuns"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{eval_id}/runs")
async def createEvalRun(eval_id: str, body: CreateEvalRunRequest) -> EvalRun:
    """Operation ID: createEvalRun"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.delete("/{eval_id}/runs/{run_id}")
async def deleteEvalRun(eval_id: str, run_id: str) -> dict:
    """Operation ID: deleteEvalRun"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{eval_id}/runs/{run_id}")
async def getEvalRun(eval_id: str, run_id: str) -> EvalRun:
    """Operation ID: getEvalRun"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{eval_id}/runs/{run_id}")
async def cancelEvalRun(eval_id: str, run_id: str) -> EvalRun:
    """Operation ID: cancelEvalRun"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{eval_id}/runs/{run_id}/output_items")
async def getEvalRunOutputItems(eval_id: str, run_id: str) -> EvalRunOutputItemList:
    """Operation ID: getEvalRunOutputItems"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{eval_id}/runs/{run_id}/output_items/{output_item_id}")
async def getEvalRunOutputItem(
    eval_id: str, run_id: str, output_item_id: str
) -> EvalRunOutputItem:
    """Operation ID: getEvalRunOutputItem"""
    raise NotImplementedError("Endpoint not yet implemented")
