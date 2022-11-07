from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from pprint import pformat, pprint
from typing import Any, Final, Optional
from uuid import UUID
import json
import logging
import sys

from gladier import GladierBaseClient, GladierBaseTool, generate_flow_definition
from globus_automate_client import FlowsClient
from globus_automate_client.flows_client import (MANAGE_FLOWS_SCOPE, RUN_FLOWS_SCOPE,
                                                 RUN_STATUS_SCOPE, VIEW_FLOWS_SCOPE)
from globus_sdk import NativeAppAuthClient, OAuthTokenResponse, RefreshTokenAuthorizer

from .client import WorkflowClient, WorkflowClientBuilder, WorkflowRun
from .settings import WorkflowSettings

logger = logging.getLogger(__name__)


def ptychodus_reconstruct(**data):
    from pathlib import Path
    from ptychodus.model import ModelArgs, ModelCore

    modelArgs = ModelArgs(
        settingsFilePath=Path(data['settings_file']),
        replacementPathPrefix=data['replacement_path_prefix'],
    )

    with ModelCore(modelArgs) as model:
        model.batchModeReconstruct()


@generate_flow_definition
class PtychodusReconstruct(GladierBaseTool):
    required_input = [
        'settings_file',
        'replacement_path_prefix',
    ]
    funcx_functions = [ptychodus_reconstruct]


@generate_flow_definition
class PtychodusClient(GladierBaseClient):
    # TODO client_id = GlobusWorkflowClient.CLIENT_ID
    gladier_tools = [
        'gladier_tools.globus.transfer.Transfer:InputData',
        PtychodusReconstruct,
        'gladier_tools.globus.transfer.Transfer:OutputData',
        #'gladier_tools.publish.Publish',
    ]


class GlobusWorkflowClient(WorkflowClient):
    CLIENT_ID: Final[str] = '5c0fb474-ae53-44c2-8c32-dd0db9965c57'

    def __init__(self, settings: WorkflowSettings, authorizerDict: dict[str, Any]) -> None:
        super().__init__()
        self._settings = settings
        self._authorizerDict = authorizerDict
        self._client = FlowsClient.new_client(client_id=self.CLIENT_ID,
                                              authorizer_callback=self._authorizerRetriever,
                                              authorizer=authorizerDict.get(MANAGE_FLOWS_SCOPE))

    def _authorizerRetriever(self, flow_url: str, flow_scope: str,
                             client_id: str) -> RefreshTokenAuthorizer:
        logger.debug(f'Searching for flow scope {flow_scope}')
        return self._authorizerDict[flow_scope]

    def listFlowRuns(self) -> list[WorkflowRun]:
        runList: list[WorkflowRun] = list()
        flowID = str(self._settings.flowID.value)
        orderings = {'start_time': 'desc'}  # ordering by start_time (descending)
        response = self._client.list_flow_runs(flow_id=flowID, orderings=orderings)
        logger.debug(f'Flow Run List: {response}')

        for runDict in response['runs']:
            runID = runDict.get('run_id', '')
            runURL = f'https://app.globus.org/runs/{runID}/logs'
            run = WorkflowRun(label=runDict.get('label', ''),
                              startTime=runDict.get('start_time', ''),
                              completionTime=runDict.get('completion_time', ''),
                              status=runDict.get('status', ''),
                              action=runDict.get('display_status', ''),
                              runID=runID,
                              runURL=runURL)
            runList.append(run)

        # FIXME display_status -> current action
        #flow_action_id = response.data['actions'][0]['action_id']
        #flow_action = flows_client.flow_action_status(flow_id, flow_scope,
        #                                              flow_action_id)
        #try:
        #    flow_step = flow_action['details']['action_statuses'][0]['state_name']
        #except:
        #    flow_step = 'None'

        #flow_status = flow_action['status']
        #scanid = re.search(r'^\d+', flow_action['label'])[0]
        #info = f"{scanid} {flow_step} {flow_status}"

        return runList

    def runFlow(self) -> None:
        flowID = str(self._settings.flowID.value)
        flowScope = None

        flowInput = {
            'input_data_transfer_source_endpoint_id':
            str(self._settings.inputDataEndpointID.value),
            'input_data_transfer_source_path':
            str(self._settings.inputDataPath.value),
            'input_data_transfer_destination_endpoint_id':
            str(self._settings.computeDataEndpointID.value),
            'input_data_transfer_destination_path':
            str(self._settings.computeDataPath.value),
            'input_data_transfer_recursive':
            False,
            'funcx_endpoint_compute':
            str(self._settings.computeEndpointID.value),
            'ptychodus_reconstruct_funcx_id':
            str(self._settings.reconstructActionID.value),
            'output_data_transfer_source_endpoint_id':
            str(self._settings.computeDataEndpointID.value),
            'output_data_transfer_source_path':
            str(self._settings.computeDataPath.value),
            'output_data_transfer_destination_endpoint_id':
            str(self._settings.outputDataEndpointID.value),
            'output_data_transfer_destination_path':
            str(self._settings.outputDataPath.value),
            'output_data_transfer_recursive':
            False,
        }

        flowInput = {'input': flowInput}
        flowLabel = 'Ptychodus'  # TODO
        flowTags = ['aps', 'ptychography']

        runResponse = self._client.run_flow(flowID,
                                            flowScope,
                                            flowInput,
                                            label=flowLabel,
                                            tags=flowTags)
        logger.info(f'Run Flow Response: {json.dumps(runResponse.data, indent=4)}')


class GlobusWorkflowClientBuilder(WorkflowClientBuilder):

    def __init__(self, settings: WorkflowSettings) -> None:
        super().__init__()
        self._settings = settings
        self._authClient: Optional[NativeAppAuthClient] = None

    def getAuthorizeURL(self) -> str:
        if self._authClient is None:
            gacVersion = version('globus-automate-client')
            logger.info(f'\tGlobus Automate Client {gacVersion}')
            self._authClient = NativeAppAuthClient(GlobusWorkflowClient.CLIENT_ID)

        FLOW_ID = str(self._settings.flowID.value)
        FLOW_ID_ = FLOW_ID.replace('-', '_')

        requestedScopes = [
            # Automate scopes
            MANAGE_FLOWS_SCOPE,
            RUN_FLOWS_SCOPE,
            RUN_STATUS_SCOPE,
            VIEW_FLOWS_SCOPE,

            # Flow scope
            f'https://auth.globus.org/scopes/{FLOW_ID}/flow_{FLOW_ID_}_user',
        ]

        logger.debug('Requested Scopes: {pformat(requestedScopes)}')

        self._authClient.oauth2_start_flow(requested_scopes=requestedScopes, refresh_tokens=True)
        return self._authClient.oauth2_get_authorize_url()

    def build(self, authCode: str) -> GlobusWorkflowClient:
        if self._authClient is None:
            raise RuntimeError('Missing AuthClient!')

        tokenResponse = self._authClient.oauth2_exchange_code_for_tokens(authCode.strip())
        logger.debug(f'Token response: {tokenResponse}')
        authorizerDict: dict[str, Any] = dict()

        for resourceServer, tokenData in tokenResponse.by_resource_server.items():
            authorizer = RefreshTokenAuthorizer(
                refresh_token=tokenData['refresh_token'],
                auth_client=self._authClient,
                access_token=tokenData['access_token'],
                expires_at=tokenData['expires_at_seconds'],
            )

            for scope in tokenData['scope'].split():
                authorizerDict[scope] = authorizer

        logger.debug('Authorizers: {pformat(authorizerDict)}')

        return GlobusWorkflowClient(self._settings, authorizerDict)


def main() -> int:
    client = PtychodusClient()

    flow_id = client.get_flow_id()
    print(f'Flow ID: {flow_id}')

    pprint(client.flow_definition)

    print('FuncX IDs:')
    funcx_ids = client.get_funcx_function_ids()

    for name, value in funcx_ids.items():
        print(f'\t{name} -> {value}')

    return 0


if __name__ == '__main__':
    sys.exit(main())