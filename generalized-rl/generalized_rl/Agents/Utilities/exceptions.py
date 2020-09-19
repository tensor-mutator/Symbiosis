__all__ = ["MissingReplayError"]

class MissingReplayError(Exception):

      def __init__(self, msg: str) -> None:
          super(MissingReplayError, self).__init__(msg)

