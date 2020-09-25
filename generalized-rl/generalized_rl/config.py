from enum import IntEnum

class config(IntEnum):

      DEFAULT: bin = 0b0000000000
      CONSOLE_SUMMARY: bin = 0b0000000001
      LOG_SUMMARY: bin = 0b0000000010
      REWARD_EVENT: bin = 0b0000000100
      LOSS_EVENT: bin = 0b0000001000
      EPSILON_EVENT: bin = 0b0000010000
      BETA_EVENT: bin = 0b0000100000
      LR_EVENT: bin = 0b0001000000
      SAVE_WEIGHTS: bin = 0b0010000000
      LOAD_WEIGHTS: bin = 0b0100000000
      NOHUP: bin = 0b1000000000
