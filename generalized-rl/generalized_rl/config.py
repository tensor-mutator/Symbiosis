from enum import IntEnum

class config(IntEnum):

      DEFAULT: bin = 0b0000
      CONSOLE_SUMMARY: bin = 0b0001
      LOG_SUMMARY: bin = 0b0010
      TENSOR_EVENT: bin = 0b0100
      SAVE_WEIGHTS: bin = 0b1000
