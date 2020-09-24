from enum import IntEnum

class config(IntEnum):

      CONSOLE_SUMMARY: bin = 0b001
      LOG_SUMMARY: bin = 0b010
      TENSOR_EVENT: bin = 0b100
