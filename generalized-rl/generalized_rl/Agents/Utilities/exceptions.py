__all__ = ["MissingReplayError"]

class MissingReplayError(Exception):

      def __init__(self, msg: str) -> None:
          super(MissingReplayError, self).__init__(msg)

class MissingRewardArtifactError(Exception):

      def __init__(self, msg: str) -> None:
          super(MissingRewardArtifactError, self).__init__(msg)

class MissingSuiteError(Exception):

      def __init__(self, msg: str) -> None:
          super(MissingSuiteError, self).__init__(msg)
