#!/bin/bash
exec python3 masks_consist.py &
exec python3 clear_dataset.py &
exec python3 squeeze.py