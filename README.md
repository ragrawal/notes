## Mac Installation Instructions

```
# install pandoc to convert markdown to pdf
brew install pandoc

# install pdflatex engine required by pandoc
brew cask install mactex


# Add pagebreak to file
sed 's/^## /\\pagebreak\ 
&/g' ~/Desktop/Pytorch/Pytorch.md > tmp.md

# Convert file to pdf
pandoc tmp.md -t latex -V geometry:paperwidth=5in -V geometry:paperheight=3in -V geometry:margin=0.5in -o pytorch.pdf
```