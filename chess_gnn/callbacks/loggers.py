from pytorch_lightning.loggers import CometLogger


class CometLoggerCallbackFactory:
    def __init__(self, project_name: str, workspace: str = 'rbiju'):
        self.project_name = project_name
        self.workspace = workspace

    def logger(self, api_key: str) -> CometLogger:
        return CometLogger(api_key=api_key, project_name=self.project_name, workspace=self.workspace)
