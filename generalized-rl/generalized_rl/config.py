from enum import IntEnum, unique

@unique
class config(IntEnum):

      DEFAULT: bin = 0b00000000000
      VERBOSE_LITE: bin = 0b00000000001
      VERBOSE_HEAVY: bin = 0b00000000010
      LOG_SUMMARY: bin = 0b00000000100
      REWARD_EVENT: bin = 0b00000001000
      LOSS_EVENT: bin = 0b00000010000
      EPSILON_EVENT: bin = 0b00000100000
      BETA_EVENT: bin = 0b00001000000
      LR_EVENT: bin = 0b00010000000
      SAVE_WEIGHTS: bin = 0b00100000000
      LOAD_WEIGHTS: bin = 0b01000000000
      NOHUP: bin = 0b10000000000

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
          print("config.NOHUP")
