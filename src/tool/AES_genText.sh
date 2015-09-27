#!/bin/bash
# Generate Random Text of given size
# Usage: ./<exec> <file size>
# An ASCII file will be generated of given file size
if [ -z "$1" ] && [ -z "$2" ]
then
    echo "Usage: ./<exec> <file size>"
else
    base64 /dev/urandom | head -c $1 > $2
fi