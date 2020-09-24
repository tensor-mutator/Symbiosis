from enum import IntEnum

class config(IntEnum):

      DEFAULT: bin = 0b00000
      CONSOLE_SUMMARY: bin = 0b00001
      LOG_SUMMARY: bin = 0b00010
      TENSOR_EVENT: bin = 0b00100
      SAVE_WEIGHTS: bin = 0b01000
      LOAD_WEIGHTS: bin = 0b10000
