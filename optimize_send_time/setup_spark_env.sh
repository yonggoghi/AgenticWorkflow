#!/bin/bash

# ============================================================================
# Spark 3.1.3 환경 구성 스크립트 (macOS)
# ============================================================================

set -e

echo "========================================="
echo "Spark 3.1.3 Environment Setup"
echo "========================================="
echo ""

# Homebrew 권한 수정
echo "1. Fixing Homebrew permissions..."
sudo chown -R $(whoami) /usr/local/Cellar /usr/local/Homebrew

# Java 11 설치
echo ""
echo "2. Installing Java 11 (OpenJDK)..."
if /usr/libexec/java_home -v 11 &>/dev/null; then
    echo "   ✓ Java 11 already installed"
else
    brew install openjdk@11
    echo "   ✓ Java 11 installed"
fi

# Java 11 심볼릭 링크 생성
echo ""
echo "3. Setting up Java 11 symlink..."
sudo ln -sfn /usr/local/opt/openjdk@11/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-11.jdk

# Scala 설치 (Spark 3.1.3은 Scala 2.12 사용)
echo ""
echo "4. Installing Scala 2.12..."
if command -v scala &>/dev/null; then
    echo "   ✓ Scala already installed"
else
    brew install scala@2.12
    echo "   ✓ Scala 2.12 installed"
fi

# Spark 경로 확인
SPARK_HOME="/Users/yongwook/spark-local/spark-3.1.3-bin-hadoop3.2"
if [ ! -d "$SPARK_HOME" ]; then
    echo ""
    echo "5. Spark 3.1.3 not found. Downloading..."
    mkdir -p /Users/yongwook/spark-local
    cd /Users/yongwook/spark-local
    curl -O https://archive.apache.org/dist/spark/spark-3.1.3/spark-3.1.3-bin-hadoop3.2.tgz
    tar -xzf spark-3.1.3-bin-hadoop3.2.tgz
    rm spark-3.1.3-bin-hadoop3.2.tgz
    echo "   ✓ Spark 3.1.3 downloaded and extracted"
else
    echo ""
    echo "5. Spark 3.1.3 already exists at $SPARK_HOME"
fi

# 환경 변수 설정 (zsh)
echo ""
echo "6. Setting up environment variables..."

# .zshrc 백업
if [ -f ~/.zshrc ]; then
    cp ~/.zshrc ~/.zshrc.backup_$(date +%Y%m%d_%H%M%S)
fi

# 기존 SPARK 설정 제거
sed -i '' '/# Spark Environment/,/# End Spark Environment/d' ~/.zshrc 2>/dev/null || true

# 새로운 설정 추가
cat >> ~/.zshrc << 'EOF'

# Spark Environment
export JAVA_HOME=$(/usr/libexec/java_home -v 11)
export SPARK_HOME=/Users/yongwook/spark-local/spark-3.1.3-bin-hadoop3.2
export SCALA_HOME=/usr/local/opt/scala@2.12
export PATH=$JAVA_HOME/bin:$SPARK_HOME/bin:$SPARK_HOME/sbin:$SCALA_HOME/bin:$PATH
export PYSPARK_PYTHON=python3

# Spark 메모리 설정
export SPARK_DRIVER_MEMORY=4g
export SPARK_EXECUTOR_MEMORY=4g

# Google OR-Tools 라이브러리 경로 (필요시)
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH
# End Spark Environment
EOF

echo "   ✓ Environment variables added to ~/.zshrc"

# Google OR-Tools 설치
echo ""
echo "7. Installing Google OR-Tools..."
brew install or-tools || echo "   ⚠ OR-Tools installation skipped (may already be installed)"

echo ""
echo "========================================="
echo "Installation Complete!"
echo "========================================="
echo ""
echo "Please run the following command to apply changes:"
echo "   source ~/.zshrc"
echo ""
echo "Or restart your terminal."
echo ""
echo "To verify installation:"
echo "   java -version"
echo "   scala -version"
echo "   spark-shell --version"
echo ""
