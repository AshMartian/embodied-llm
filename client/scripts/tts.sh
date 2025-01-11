#!/bin/bash

echo "$1" | \
  ~/.local/bin/piper --model ./checkpoints/en_US-amy-medium.onnx --output-raw | \
  aplay -r 22050 -f S16_LE -t raw -