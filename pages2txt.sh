#!/bin/bash
# Convert all .pages files in the current folder to .txt using Pages

for f in *.pages; do
    base=$(basename "$f" .pages)
    osascript -e "tell application \"Pages\"
        open POSIX file \"$(pwd)/$f\"
        set outFile to POSIX file \"$(pwd)/$base.txt\"
        export front document to outFile as unformatted text
        close front document saving no
    end tell"
    echo "✅ Converted $f → $base.txt"
done

