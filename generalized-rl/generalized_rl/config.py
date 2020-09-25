from enum import IntEnum

class config(IntEnum):

      DEFAULT: bin = 0b000000
      CONSOLE_SUMMARY: bin = 0b000001
      LOG_SUMMARY: bin = 0b000010
      TENSOR_EVENT: bin = 0b000100
      SAVE_WEIGHTS: bin = 0b001000
      LOAD_WEIGHTS: bin = 0b010000
      NOHUP: bin = 0b100000
