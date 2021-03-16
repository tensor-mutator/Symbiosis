__all__ = ["MissingReplayError", "MissingRewardArtifactError", "MissingSuiteError", "AgentInterrupt", "HyperparamsMismatchError",
           "UnregisteredSchemeError", "TreeError"]

class MissingReplayError(Exception):

      def __init__(self, msg: str) -> None:
          super(MissingReplayError, self).__init__(msg)

class TreeError(Exception):

      def __init__(self, msg: str) -> None:
          super(TreeError, self).__init__(msg)

class MissingRewardArtifactError(Exception):

      def __init__(self, msg: str) -> None:
          super(MissingRewardArtifactError, self).__init__(msg)

class MissingSuiteError(Exception):

      def __init__(self, msg: str) -> None:
          super(MissingSuiteError, self).__init__(msg)

class AgentInterrupt(Exception):

      def __init__(self, msg: str) -> None:
          super(AgentInterrupt, self).__init__(msg)

class HyperparamsMismatchError(Exception):

      def __init__(self, msg: str) -> None:
          super(HyperparamsMismatchError, self).__init__(msg)

class MissingHyperparamsError(Exception):

      def __init__(self, msg: str) -> None:
          super(MissingHyperparamsError, self).__init__(msg)

class UnregisteredSchemeError(Exception):

      def __init__(self, msg: str) -> None:
          super(UnregisteredSchemeError, self).__init__(msg)
