from __future__ import annotations

import json
import logging
import os
import queue
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping
from importlib.metadata import version
from pprint import pformat
from typing import Final, Optional, Union

import fair_research_login
import gladier
import gladier.managers
import globus_sdk
from gladier.tools import Compute, Transfer, TransferItem
from gladier.tools.globus import ComputeFunctionType
from gladier_tools.publish import Publishv2State

# from .authorizer import WorkflowAuthorizer
# from .executor import WorkflowExecutor
# from .status import WorkflowStatus, WorkflowStatusRepository

logger = logging.getLogger(__name__)

AuthorizerTypes = Union[
    globus_sdk.AccessTokenAuthorizer, globus_sdk.RefreshTokenAuthorizer
]
ScopeAuthorizerMapping = Mapping[str, AuthorizerTypes]

PTYCHODUS_CLIENT_ID: Final[str] = "5c0fb474-ae53-44c2-8c32-dd0db9965c57"


def ptychodus_reconstruct(
    ptychodus_restart_file: str,
    ptychodus_settings_file: str,
    ptychodus_results_file: str,
    **kwargs,
) -> None:
    from pathlib import Path

    from ptychodus.model import ModelArgs, ModelCore

    modelArgs = ModelArgs(
        restartFilePath=Path(ptychodus_restart_file),
        settingsFilePath=Path(ptychodus_settings_file),
    )

    resultsFilePath = Path(ptychodus_results_file)

    with ModelCore(modelArgs) as model:
        model.batchModeReconstruct(resultsFilePath)


def ptychodus_reconstruct_placeholder(
    ptychodus_restart_file: str,
    ptychodus_settings_file: str,
    ptychodus_results_file: str,
    **kwargs,
) -> None:
    from pathlib import Path

    from ptychodus.model import ModelArgs, ModelCore

    """
    modelArgs = ModelArgs(
        restartFilePath=Path(ptychodus_restart_file),
        settingsFilePath=Path(ptychodus_settings_file),
    )
    """

    resultsFilePath = Path(ptychodus_results_file)

    with open(resultsFilePath, "w") as f:
        f.write("Here's the result")


def create_flow_definition() -> gladier.GladierBaseState:
    class PtychodusReconstructState(Compute):
        function_to_call: ComputeFunctionType = ptychodus_reconstruct_placeholder
        wait_time = 7200
        ptychodus_restart_file: str = "$.input.ptychodus_restart_file"
        ptychodus_settings_file: str = "$.input.ptychodus_settings_file"
        ptychodus_results_file: str = "$.input.ptychodus_results_file"

        def get_flow_definition(self) -> gladier.JSONObject:
            self.set_call_params_from_self_model(
                {
                    "ptychodus_restart_file",
                    "ptychodus_settings_file",
                    "ptychodus_results_file",
                }
            )
            return super().get_flow_definition()

    reconstruct = PtychodusReconstructState()
    publish = Publishv2State(transfer_enabled=False, ingest_enabled=True)
    reconstruct.next(publish)
    return reconstruct
    """
    return (
        Transfer(
            state_name="TransferInputData",
            source_endpoint_id="$.input.input_data_transfer_source_endpoint_id",
            destination_endpoint_id="$.input.input_data_transfer_destination_endpoint_id",
            transfer_items=[
                TransferItem(
                    source_path="$.input.input_data_transfer_source_path",
                    destination_path="$.input.input_data_transfer_destination_path",
                    recursive="$.input.input_data_transfer_recursive",
                )
            ],
        )
        .next(PtychodusReconstructState())
        .next(
            Transfer(
                state_name="TransferOutputData",
                source_endpoint_id="$.input.output_data_transfer_source_endpoint_id",
                destination_endpoint_id="$.input.output_data_transfer_destination_endpoint_id",
                transfer_items=[
                    TransferItem(
                        source_path="$.input.output_data_transfer_source_path",
                        destination_path="$.input.output_data_transfer_destination_path",
                        recursive="$.input.output_data_transfer_recursive",
                    )
                ],
            )
        )
    )
    """


def main():
    root_state = create_flow_definition()
    client2 = gladier.GladierClient(
        flow_definition=root_state.get_flow_definition(),
    )
    flow_def2 = client2.get_flow_definition()
    print(f"DEBUG {(flow_def2)=}")

    jims_laptop_globus_compute_endpoint = "3f14fbd6-80e7-4ae0-8f55-766f18aff24f"
    jims_alcf_globus_compute_endpoint = "a648215f-ff50-4b36-81e3-ea002bdd8047"

    def to_local_path(globus_path: str) -> str:
        return f"/eagle{globus_path}"

    eagle_globus_collection_id = "05d2c76a-e867-4f67-aa57-76edeb0beda0"
    source_dataset_root = "/APSDataAnalysis/PTYCHO/test_data/fly001/"
    work_dataset_root = "/APSDataAnalysis/pruyne/ptychodus/data/fly001/"

    flow_input = {
        "input_data_transfer_source_endpoint_id": eagle_globus_collection_id,
        "input_data_transfer_destination_endpoint_id": eagle_globus_collection_id,
        "input_data_transfer_source_path": source_dataset_root,
        "input_data_transfer_destination_path": work_dataset_root,
        "input_data_transfer_recursive": False,
        "ptychodus_restart_file": None,
        "ptychodus_settings_file": to_local_path(f"{source_dataset_root}ptychodus.ini"),
        # "ptychodus_results_file": to_local_path(f"{work_dataset_root}run_results.out"),
        "ptychodus_results_file": "/home/pruyne/run_results.out",
        "publishv2": {
            "dataset": "/home/pruyne/run_results.out",
            "my-globus-search-index-uuid": "93e343cc-b555-4d60-9aab-80ff191a8abb",
            "my-source-globus-collection": eagle_globus_collection_id,
            "my-destination-globus-collection": eagle_globus_collection_id,
            "destination": "/home/pruyne",
            "ingest_enabled": True,
        },
        "compute_endpoint": jims_laptop_globus_compute_endpoint,
        "globus_compute_endpoint_non_compute": jims_laptop_globus_compute_endpoint,
    }
    run_info = client2.run_flow(flow_input={"input": flow_input})
    print(f"DEBUG {(run_info)=}")


if __name__ == "__main__":
    main()
