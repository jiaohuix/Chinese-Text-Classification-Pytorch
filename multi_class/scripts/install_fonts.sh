#!/bin/bash

apt-get update  -y
apt install libpango1.0-dev -y
# apt install libpango1.0-0 --fix-missing
 
# Clone the repository
# git clone https://github.com/tracyone/program_font

# Change directory to the cloned repository
cd ./program_font/

# Install the font (assuming install.sh handles the installation correctly)
./install.sh

# Get the matplotlib font directory using python
FONT_DIR=$(python -c "import matplotlib; print(matplotlib.matplotlib_fname().replace('matplotlibrc', 'fonts'))")

# Check if the font directory exists
if [ ! -d "$FONT_DIR" ]; then
  echo "Error: Matplotlib font directory not found: $FONT_DIR"
  exit 1
fi

# Copy the simhei.ttf font file to the matplotlib font directory
cp simhei.ttf "$FONT_DIR"

# Remove the matplotlib cache directory
rm -rf ~/.cache/matplotlib

echo "Font installation complete."
echo "Please restart your Python environment or kernel for the changes to take effect."
