#!/bin/bash

# Create directories for packages
mkdir -p ./macos_packages
mkdir -p ./linux_packages

# List of required packages based on imports
PACKAGES=(
    # "flask"
    # "flask-cors"
    # "pandas"
    # "openai"
    # "rapidfuzz"
    # "langchain-anthropic"
    # "langchain-openai"
    # "langchain-core"
    # "scikit-learn"
    # "python-dotenv"
    "kiwipiepy==0.21.0"
)

# Download packages for macOS
echo "Downloading packages for macOS..."
for package in "${PACKAGES[@]}"; do
    echo "Downloading $package for macOS..."
    pip download $package --dest ./macos_packages
    if [ $? -ne 0 ]; then
        echo "Failed to download $package for macOS"
    fi
done

# Download packages for Linux
echo "Downloading packages for Linux..."
for package in "${PACKAGES[@]}"; do
    echo "Downloading $package for Linux..."
    
    # Special handling for kiwipiepy due to dependency conflicts
    if [[ "$package" == kiwipiepy* ]]; then
        echo "Using alternative approach for kiwipiepy on Linux..."
        # Download without strict dependency resolution
        pip download $package \
            --platform manylinux2014_x86_64 \
            --only-binary=:all: \
            --python-version 312 \
            --implementation cp \
            --abi cp312 \
            --no-deps \
            --dest ./linux_packages
        
        # Download dependencies separately for kiwipiepy 0.21.0
        echo "Downloading kiwipiepy dependencies for Linux..."
        # Download source distribution for kiwipiepy_model since no wheel is available for Linux
        pip download "kiwipiepy_model>=0.21,<0.22" \
            --no-binary kiwipiepy_model \
            --dest ./linux_packages 2>/dev/null || echo "Note: kiwipiepy_model source distribution downloaded"
        
        pip download numpy \
            --platform manylinux2014_x86_64 \
            --only-binary=:all: \
            --python-version 312 \
            --implementation cp \
            --abi cp312 \
            --dest ./linux_packages
            
        pip download tqdm \
            --platform manylinux2014_x86_64 \
            --only-binary=:all: \
            --python-version 312 \
            --implementation cp \
            --abi cp312 \
            --dest ./linux_packages
    else
        # Standard download for other packages
        pip download $package \
            --platform manylinux2014_x86_64 \
            --only-binary=:all: \
            --python-version 312 \
            --implementation cp \
            --abi cp312 \
            --dest ./linux_packages
    fi
    
    if [ $? -ne 0 ]; then
        echo "Failed to download $package for Linux"
    fi
done

# Create zip files
echo "Creating zip files..."
# zip -r macos_packages.zip ./macos_packages
if [ -d "./linux_packages" ] && [ "$(ls -A ./linux_packages)" ]; then
    zip -r linux_packages.zip ./linux_packages
    echo "Linux packages: linux_packages.zip"
else
    echo "No Linux packages to zip"
fi

echo "Download and packaging complete!"
# echo "macOS packages: macos_packages.zip"
# echo "Linux packages: linux_packages.zip" 