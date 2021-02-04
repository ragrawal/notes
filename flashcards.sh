# 
# Usage: ./flashcards input_file output_file
#
sed 's/^## /\\pagebreak&/g' $1 > /tmp/tmp.md

# Convert file to pdf
pandoc /tmp/tmp.md \
	-t latex \
	-V geometry:paperwidth=5in \
	-V geometry:paperheight=3in \
	-V geometry:left=0.5in \
	-V geometry:right=0.3in \
	-V geometry:top=0.3in \
	-V geometry:bottom=0.3in \
	-o $2

# remove temporary file
rm /tmp/tmp.md 

# Open output file
open $2