#!/bin/bash

echo "🚀 Starting home directory cleanup..."

# 1️⃣ Trash and Downloads
echo "Cleaning Trash and Downloads..."
rm -rf ~/.local/share/Trash/*
rm -rf ~/Downloads/*

# 2️⃣ Node.js cache
echo "Cleaning npm cache..."
npm cache clean --force 2>/dev/null
rm -rf ~/.npm

# 3️⃣ Flutter / Dart cache
echo "Cleaning Flutter and Dart caches..."
rm -rf ~/.pub-cache
rm -rf ~/.dartServer
rm -rf ~/.dart-tool
rm -rf ~/Documents/flutter/bin/cache
# Optional: remove .git history of Flutter SDK if you don't need it
# rm -rf ~/Documents/flutter/.git

# 4️⃣ VSCode caches
echo "Cleaning VSCode caches..."
rm -rf ~/.vscode/cache
rm -rf ~/.vscode-insiders/cache

# 5️⃣ Old VSCode binaries and installers
echo "Removing old VSCode binaries and installers..."
rm -f ~/vscode*.tar.gz
rm -f ~/Downloads/code_*.deb
# Remove stable VSCode if not used
rm -f ~/VSCode-insiders/code

# 6️⃣ General caches
echo "Removing general caches..."
rm -rf ~/.cache/*

# 7️⃣ Summary
echo "✅ Cleanup done. Current home usage:"
du -sh ~
