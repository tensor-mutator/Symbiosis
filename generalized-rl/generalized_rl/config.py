from enum import IntEnum, unique

@unique
class config(IntEnum):

      DEFAULT: bin = 0b000000000000
      VERBOSE_LITE: bin = 0b000000000001
      VERBOSE_HEAVY: bin = 0b000000000010
      LOG_SUMMARY: bin = 0b000000000100
      REWARD_EVENT: bin = 0b000000001000
      LOSS_EVENT: bin = 0b000000010000
      EPSILON_EVENT: bin = 0b000000100000
      BETA_EVENT: bin = 0b000001000000
      LR_EVENT: bin = 0b000010000000
      SAVE_WEIGHTS: bin = 0b000100000000
      LOAD_WEIGHTS: bin = 0b001000000000
      SAVE_FRAMES: bin = 0b010000000000
      NOHUP: bin = 0b100000000000

      @staticmethod
      def show() -> None:
          print("config.DEFAULT")
          print("config.VERBOSE_LITE")
          print("config.VERBOSE_HEAVY")
          print("config.LOG_SUMMARY")
          print("config.REWARD_EVENT")
          print("config.LOSS_EVENT")
          print("config.EPSILON_EVENT")
          print("config.BETA_EVENT")
          print("config.LR_EVENT")
          print("config.SAVE_WEIGHTS")
          print("config.LOAD_WEIGHTS")
          print("config.SAVE_FRAMES")
          print("config.NOHUP")
